use super::Agent;
use crate::logging;
use crate::message::{Message, ToolDefinition};

impl Agent {
    pub(super) fn log_prompt_prefix_accounting(
        &self,
        split: &crate::prompt::SplitSystemPrompt,
        tools: &[ToolDefinition],
    ) {
        let system_tokens = split.estimated_tokens();
        let tool_tokens = ToolDefinition::aggregate_prompt_token_estimate(tools);
        let prefix_tokens = system_tokens + tool_tokens;
        logging::info(&format!(
            "Prompt prefix estimate: total={} tokens (system={} tools={})",
            prefix_tokens, system_tokens, tool_tokens
        ));
    }

    pub(super) fn build_memory_prompt_nonblocking_shared(
        &self,
        messages: std::sync::Arc<[Message]>,
        _memory_event_tx: Option<crate::memory::MemoryEventSink>,
    ) -> Option<crate::memory::PendingMemory> {
        if !self.memory_enabled {
            return None;
        }

        let session_id = &self.session.id;

        let fresh_user_turn = crate::message::ends_with_fresh_user_turn(&messages);
        let pending = if fresh_user_turn {
            crate::memory::take_pending_memory(session_id)
        } else {
            None
        };

        // Use the persistent memory-agent pipeline as the single source of truth.
        // Running both this and the legacy MemoryManager background retrieval path
        // can prepare overlapping pending prompts for the same turn, which makes
        // memory injection feel overly aggressive.
        // Relevance results are consumed only at the start of a fresh user turn.
        // Enqueuing again after every tool result runs the local embedding model
        // for each provider continuation without creating an additional injection
        // opportunity. One update per user turn keeps memory current while avoiding
        // redundant 512-token inference during tool-heavy agent loops.
        if fresh_user_turn {
            crate::memory_agent::update_context_sync_with_dir(
                session_id,
                messages,
                self.session.working_dir.clone(),
            );
        }

        pending
    }

    fn append_current_turn_system_reminder(&self, split: &mut crate::prompt::SplitSystemPrompt) {
        let Some(reminder) = self
            .current_turn_system_reminder
            .as_ref()
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
        else {
            return;
        };

        if !split.dynamic_part.is_empty() {
            split.dynamic_part.push_str("\n\n");
        }
        split.dynamic_part.push_str("# System Reminder\n\n");
        split.dynamic_part.push_str(reminder);
    }

    /// Build split system prompt for better caching
    /// Returns static (cacheable) and dynamic (not cached) parts separately
    pub(super) fn build_system_prompt_split(
        &self,
        memory_prompt: Option<&str>,
    ) -> crate::prompt::SplitSystemPrompt {
        if let Some(ref override_prompt) = self.system_prompt_override {
            return crate::prompt::SplitSystemPrompt {
                static_part: override_prompt.clone(),
                dynamic_part: String::new(),
            };
        }

        let skills = self.current_skills_snapshot();
        let skill_prompt = self
            .active_skill
            .as_ref()
            .and_then(|name| skills.get(name).map(|skill| skill.get_prompt().to_string()));

        let available_skills: Vec<crate::prompt::SkillInfo> = self
            .current_skills_snapshot()
            .list()
            .iter()
            .map(|skill| crate::prompt::SkillInfo {
                name: skill.name.clone(),
                description: skill.description.clone(),
            })
            .collect();

        let working_dir = self
            .session
            .working_dir
            .as_ref()
            .map(std::path::PathBuf::from);

        // Pin the file-backed static inputs once per session so mid-session
        // edits to system-prompt.md / AGENTS.md / prompt-overlay.md /
        // preferred-tools.md cannot silently invalidate the provider prompt
        // cache. Drift is surfaced as a one-time warning instead.
        let files = self
            .static_prompt_files
            .get_or_init(|| crate::prompt::StaticPromptFiles::load(working_dir.as_deref()));
        let changes = files.detect_changes(working_dir.as_deref());
        if !changes.is_empty()
            && !self
                .static_prompt_change_warned
                .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            let detail = changes
                .iter()
                .map(|change| {
                    format!(
                        "{} ({} -> {} bytes)",
                        change.label, change.pinned_bytes, change.current_bytes
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            logging::warn(&format!(
                "SYSTEM_PROMPT_PREFIX_CHANGED: {}; keeping pinned prefix for this session (restart to apply)",
                detail
            ));
        }

        let (mut split, _context_info) = crate::prompt::build_system_prompt_split_from_files(
            files,
            skill_prompt.as_deref(),
            &available_skills,
            self.session.is_canary,
            memory_prompt,
            working_dir.as_deref(),
        );

        self.append_current_turn_system_reminder(&mut split);
        crate::prompt::append_swarm_effort_directive(
            &mut split,
            self.provider.reasoning_effort().as_deref(),
        );

        split
    }

    /// Non-blocking memory prompt - takes pending result and spawns check for next turn
    #[cfg(test)]
    pub(super) fn build_memory_prompt_nonblocking(
        &self,
        messages: &[Message],
        _memory_event_tx: Option<crate::memory::MemoryEventSink>,
    ) -> Option<crate::memory::PendingMemory> {
        self.build_memory_prompt_nonblocking_shared(messages.to_vec().into(), _memory_event_tx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::Arc;

    struct MockProvider;

    #[async_trait]
    impl crate::provider::Provider for MockProvider {
        async fn complete(
            &self,
            _messages: &[Message],
            _tools: &[ToolDefinition],
            _system: &str,
            _resume_session_id: Option<&str>,
        ) -> anyhow::Result<crate::provider::EventStream> {
            Err(anyhow::anyhow!(
                "mock provider must not complete in prompt tests"
            ))
        }

        fn name(&self) -> &str {
            "mock"
        }

        fn model(&self) -> String {
            "mock-model".to_string()
        }

        fn fork(&self) -> Arc<dyn crate::provider::Provider> {
            Arc::new(MockProvider)
        }
    }

    fn real_agent_with_working_dir(working_dir: String) -> Agent {
        let provider: Arc<dyn crate::provider::Provider> = Arc::new(MockProvider);
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let _guard = rt.enter();
        let registry = rt.block_on(crate::tool::Registry::new(provider.clone()));
        let mut session =
            crate::session::Session::create_with_id("prompt-pin-test".to_string(), None, None);
        session.model = Some("mock-model".to_string());
        session.working_dir = Some(working_dir);
        Agent::new_with_session(provider, registry, session, None)
    }

    /// Real-Agent integration boundary: the session pin must survive across
    /// turns of the actual `Agent::build_system_prompt_split` path, ignoring a
    /// mid-session overlay edit while surfacing drift exactly once.
    #[test]
    fn agent_build_system_prompt_split_pins_static_files_per_session() {
        let _guard = crate::storage::lock_test_env();
        let prev_home = std::env::var_os("JCODE_HOME");
        let temp = tempfile::TempDir::new().unwrap();
        crate::env::set_var("JCODE_HOME", temp.path());
        let project_dir = tempfile::TempDir::new().unwrap();
        std::fs::create_dir_all(project_dir.path().join(".jcode")).unwrap();
        std::fs::write(
            project_dir.path().join(".jcode/prompt-overlay.md"),
            "overlay v1",
        )
        .unwrap();

        let mut agent = real_agent_with_working_dir(project_dir.path().display().to_string());
        assert!(
            !agent
                .static_prompt_change_warned
                .load(std::sync::atomic::Ordering::Relaxed)
        );

        // Initialize the real logger against the sandboxed JCODE_HOME so the
        // drift warning is observable in the actual rotated log file.
        crate::logging::init();

        let split_a = agent.build_system_prompt_split(None);
        assert!(split_a.static_part.contains("overlay v1"));
        assert!(
            !agent
                .static_prompt_change_warned
                .load(std::sync::atomic::Ordering::Relaxed)
        );
        // The pin is installed and no drift existed yet.
        let files = agent.static_prompt_files.get().expect("pin installed");
        assert!(files.detect_changes(Some(project_dir.path())).is_empty());

        // Mid-session edit by an external watcher.
        std::fs::write(
            project_dir.path().join(".jcode/prompt-overlay.md"),
            "overlay v2 with substantially longer content",
        )
        .unwrap();

        let split_b = agent.build_system_prompt_split(None);
        assert_eq!(
            split_a.static_part, split_b.static_part,
            "real Agent must keep serving the session-start static part"
        );
        assert!(
            !split_b.static_part.contains("v2"),
            "pinned static part must ignore the mid-session edit"
        );
        assert!(
            agent
                .static_prompt_change_warned
                .load(std::sync::atomic::Ordering::Relaxed),
            "drift must have been surfaced once"
        );

        // A third turn must NOT warn again (one-shot guard), and the pin is
        // still serving the original bytes while drift remains detectable.
        let split_c = agent.build_system_prompt_split(None);
        assert_eq!(split_a.static_part, split_c.static_part);
        assert!(!files.detect_changes(Some(project_dir.path())).is_empty());

        // The drift warning must have reached the real rotated log file
        // exactly once, with the byte delta, through the real logger.
        let log_dir = crate::storage::logs_dir().expect("log dir");
        let log = std::fs::read_to_string(log_dir.join(format!(
            "jcode-{}.log",
            chrono::Local::now().format("%Y-%m-%d")
        )))
        .expect("log file readable");
        let warn_count = log.matches("SYSTEM_PROMPT_PREFIX_CHANGED").count();
        assert_eq!(
            warn_count, 1,
            "expected exactly one SYSTEM_PROMPT_PREFIX_CHANGED in real log, got {warn_count}\n{log}"
        );
        assert!(log.contains("prompt-overlay.md"));
        assert!(log.contains("keeping pinned prefix for this session"));

        if let Some(prev_home) = prev_home {
            crate::env::set_var("JCODE_HOME", prev_home);
        } else {
            crate::env::remove_var("JCODE_HOME");
        }
    }
}
