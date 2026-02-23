package agent

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sipeed/picoclaw/pkg/bus"
	"github.com/sipeed/picoclaw/pkg/config"
	"github.com/sipeed/picoclaw/pkg/providers"
	"github.com/sipeed/picoclaw/pkg/session"
	"github.com/sipeed/picoclaw/pkg/tools"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockSummarizationProvider returns a predictable summary for testing
type mockSummarizationProvider struct {
	summary        string
	summarizeCount int
	failOnCall     int
	failError      error
}

func (m *mockSummarizationProvider) Chat(
	ctx context.Context,
	messages []providers.Message,
	tools []providers.ToolDefinition,
	model string,
	opts map[string]any,
) (*providers.LLMResponse, error) {
	m.summarizeCount++
	if m.summarizeCount == m.failOnCall {
		return nil, m.failError
	}
	return &providers.LLMResponse{
		Content:   m.summary,
		ToolCalls: []providers.ToolCall{},
	}, nil
}

func (m *mockSummarizationProvider) GetDefaultModel() string {
	return "mock-summarizer"
}

// TestRollingSummary_AppendOnly verifies that summaries are appended in chronological order
// (oldest at top, newest at bottom) with double newline separators.
func TestRollingSummary_AppendOnly(t *testing.T) {
	tmpDir := t.TempDir()
	cfg, agent, cleanup := setupRollingSummaryTest(t, tmpDir)
	defer cleanup()

	// Force very low token threshold to guarantee compression
	agent.CompressionCfg.ChunkSizeTokens = 1

	// Create a mock provider that returns distinct summaries for each compression
	summaries := []string{
		"First summary: user asked about weather, bot responded with forecast",
		"Second summary: user asked about news, bot provided headlines",
		"Third summary: user requested joke, bot told a pun",
	}
	summaryIndex := 0
	provider := &mockSummarizationProvider{
		summary: func() string {
			if summaryIndex < len(summaries) {
				s := summaries[summaryIndex]
				summaryIndex++
				return s
			}
			return "Default summary"
		}(),
	}
	agent.Provider = provider

	// Use agent-scoped session key to bypass routing override
	sessionKey := "agent:main:test1"
	msgs1 := []testMessage{
		{Role: "user", Content: "What's the weather?"},
		{Role: "assistant", Content: "It's sunny today."},
		{Role: "user", Content: "What's the news?"},
	}
	addTestMessages(agent, sessionKey, msgs1...)

	// Create AgentLoop and trigger compression via processMessage
	msgBus := bus.NewMessageBus()
	al := NewAgentLoop(cfg, msgBus, provider)
	al.registry.agents["main"] = agent

	ctx := context.Background()
	testMsg := bus.InboundMessage{
		Channel:    "test",
		SenderID:   "user",
		ChatID:     "chat1",
		Content:    "Tell me something",
		SessionKey: sessionKey, // use agent-scoped key
	}

	_, err := al.processMessage(ctx, testMsg)
	require.NoError(t, err)

	// After first compression, RollingSummary should contain the first summary
	rollingSummary1 := agent.Sessions.GetRollingSummary(sessionKey)
	assert.NotEmpty(t, rollingSummary1, "RollingSummary should not be empty after first compression")
	assert.Contains(t, rollingSummary1, summaries[0])
	assert.NotContains(t, rollingSummary1, summaries[1])
	assert.NotContains(t, rollingSummary1, summaries[2])

	// Verify format: timestamp at start of entry
	lines := strings.Split(rollingSummary1, "\n")
	assert.Regexpf(t, `^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\]$`, lines[0], "timestamp should be in [YYYY-MM-DD HH:MM] format")

	// Reset summarizer for second compression
	provider.summary = summaries[1]

	// Add more messages to trigger second compression
	msgs2 := []testMessage{
		{Role: "user", Content: "Another question?"},
		{Role: "assistant", Content: "Here's an answer."},
		{Role: "user", Content: "One more?"},
	}
	addTestMessages(agent, sessionKey, msgs2...)

	_, err = al.processMessage(ctx, testMsg)
	require.NoError(t, err)

	// After second compression, RollingSummary should contain both summaries in order
	rollingSummary2 := agent.Sessions.GetRollingSummary(sessionKey)
	assert.Contains(t, rollingSummary2, summaries[0])
	assert.Contains(t, rollingSummary2, summaries[1])

	// Verify chronological order: first summary appears before second
	idx1 := strings.Index(rollingSummary2, summaries[0])
	idx2 := strings.Index(rollingSummary2, summaries[1])
	assert.True(t, idx1 < idx2, "first summary should appear before second (oldest at top)")

	// Verify double newline separator between entries
	assert.Contains(t, rollingSummary2, "\n\n")
}

// TestRollingSummary_OnlyRawMessages verifies that summarization uses only user+assistant
// messages, excluding system, tool messages, and any RollingSummary itself.
func TestRollingSummary_OnlyRawMessages(t *testing.T) {
	tmpDir := t.TempDir()
	cfg, agent, cleanup := setupRollingSummaryTest(t, tmpDir)
	defer cleanup()

	// Force very low token threshold to guarantee compression
	agent.CompressionCfg.ChunkSizeTokens = 1

	// Track what messages were sent to the provider during summarization
	var sentMessages []providers.Message
	provider := &mockSummarizationProvider{
		summary: "Summary of user and assistant messages only",
	}
	recordingProvider := createRecordingProvider(provider, &sentMessages)
	agent.Provider = recordingProvider

	sessionKey := "agent:main:test2"
	// Add messages including system and tool messages to the session
	msgs := []testMessage{
		{Role: "system", Content: "You are a helpful assistant"},
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi there!"},
		{Role: "user", Content: "How are you?"},
		{Role: "assistant", Content: "I'm fine!"},
	}
	addTestMessages(agent, sessionKey, msgs...)

	// Create AgentLoop and trigger compression
	msgBus := bus.NewMessageBus()
	al := NewAgentLoop(cfg, msgBus, recordingProvider)
	al.registry.agents["main"] = agent

	ctx := context.Background()
	al.processMessage(ctx, bus.InboundMessage{
		Channel:    "test",
		SenderID:   "user",
		ChatID:     "chat1",
		Content:    "Go",
		SessionKey: sessionKey,
	})

	// Verify the summarization prompt only contained user+assistant messages
	require.Len(t, sentMessages, 1, "should have made one summarization call")
	summarizationInput := sentMessages[0]
	// The summarization input is a single user message containing concatenated transcript
	assert.Equal(t, "user", summarizationInput.Role)
	assert.Contains(t, summarizationInput.Content, "Hello")
	assert.Contains(t, summarizationInput.Content, "Hi there!")
	assert.Contains(t, summarizationInput.Content, "How are you?")
	assert.Contains(t, summarizationInput.Content, "I'm fine!")
	// Should NOT contain system message
	assert.NotContains(t, summarizationInput.Content, "You are a helpful assistant")
}

// TestRollingSummary_Persistence verifies RollingSummary survives session restart.
func TestRollingSummary_Persistence(t *testing.T) {
	tmpDir := t.TempDir()

	// First session: create and compress
	cfg1, agent1, cleanup1 := setupRollingSummaryTest(t, tmpDir)
	defer cleanup1()

	provider1 := &mockSummarizationProvider{summary: "Persistent summary"}
	agent1.Provider = provider1

	agent1.CompressionCfg.ChunkSizeTokens = 1

	sessionKey := "agent:main:persist"
	msgs := []testMessage{
		{Role: "user", Content: "Save this"},
		{Role: "assistant", Content: "Will do"},
	}
	addTestMessages(agent1, sessionKey, msgs...)

	// Trigger compression
	msgBus1 := bus.NewMessageBus()
	al1 := NewAgentLoop(cfg1, msgBus1, provider1)
	al1.registry.agents["main"] = agent1
	ctx := context.Background()
	al1.processMessage(ctx, bus.InboundMessage{
		Channel:    "test",
		SenderID:   "user",
		ChatID:     "chat1",
		Content:    "Trigger",
		SessionKey: sessionKey,
	})

	summary1 := agent1.Sessions.GetRollingSummary(sessionKey)
	require.NotEmpty(t, summary1)

	// Create a new SessionManager from same storage to verify persistence
	sessionsDir := filepath.Join(tmpDir, "workspace", "sessions")
	sm2 := session.NewSessionManager(sessionsDir)

	// Use GetRollingSummary to check loaded value
	summary2 := sm2.GetRollingSummary(sessionKey)
	assert.Equal(t, summary1, summary2, "RollingSummary should persist across restart")
}

// TestRollingSummary_InSystemPrompt verifies the RollingSummary appears in the
// system prompt under the "## Memory" section.
func TestRollingSummary_InSystemPrompt(t *testing.T) {
	tmpDir := t.TempDir()
	cfg, agent, cleanup := setupRollingSummaryTest(t, tmpDir)
	defer cleanup()

	sessionKey := "agent:main:test3"

	// Set a rolling summary on the session
	expectedSummary := "[2025-02-22 10:30]\nSummary about weather\n\n[2025-02-22 11:00]\nSummary about news"
	// Ensure the session exists before setting RollingSummary
	agent.Sessions.GetOrCreate(sessionKey)
	agent.Sessions.SetRollingSummary(sessionKey, expectedSummary)

	// Create a provider that records the messages sent to it
	var sentMessages []providers.Message
	provider := &mockSummarizationProvider{summary: "Response"}
	recProvider := createRecordingProvider(provider, &sentMessages)
	agent.Provider = recProvider

	// Create AgentLoop and trigger processing
	msgBus := bus.NewMessageBus()
	al := NewAgentLoop(cfg, msgBus, recProvider)
	al.registry.agents["main"] = agent

	ctx := context.Background()
	al.processMessage(ctx, bus.InboundMessage{
		Channel:    "test",
		SenderID:   "user",
		ChatID:     "chat1",
		Content:    "Hello",
		SessionKey: sessionKey,
	})

	// The provider should have been called with the full message list including system prompt
	require.NotEmpty(t, sentMessages, "provider should have been called with messages")
	// Find the system message
	var systemPrompt string
	for _, msg := range sentMessages {
		if msg.Role == "system" {
			systemPrompt = msg.Content
			break
		}
	}
	assert.NotEmpty(t, systemPrompt, "should have a system prompt")

	// Verify RollingSummary is included in the system prompt
	assert.Contains(t, systemPrompt, "## Memory")
	assert.Contains(t, systemPrompt, "**Running summary:**")
	assert.Contains(t, systemPrompt, expectedSummary)
}

// TestRollingSummary_Format verifies timestamp format and plain text formatting.
func TestRollingSummary_Format(t *testing.T) {
	tmpDir := t.TempDir()
	cfg, agent, cleanup := setupRollingSummaryTest(t, tmpDir)
	defer cleanup()

	agent.CompressionCfg.ChunkSizeTokens = 1
	agent.Provider = &mockSummarizationProvider{summary: "Test summary content"}

	sessionKey := "agent:main:format"
	msgs := []testMessage{
		{Role: "user", Content: "Check format"},
		{Role: "assistant", Content: "Will do"},
	}
	addTestMessages(agent, sessionKey, msgs...)

	al := createTestAgentLoop(t, cfg, agent)
	al.registry.agents["main"] = agent

	ctx := context.Background()
	al.processMessage(ctx, bus.InboundMessage{
		Channel:    "test",
		SenderID:   "user",
		ChatID:     "chat1",
		Content:    "x",
		SessionKey: sessionKey,
	})

	summary := agent.Sessions.GetRollingSummary(sessionKey)
	assert.NotEmpty(t, summary, "RollingSummary should not be empty")
	lines := strings.Split(summary, "\n")

	// First line should be timestamp
	assert.Regexpf(t, `^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\]$`, lines[0],
		"timestamp format should be [YYYY-MM-DD HH:MM] (Go reference time 2006-01-02 15:04)")

	// Second line should be the summary content (no extra formatting)
	assert.Len(t, lines, 2, "should have exactly two lines: timestamp and summary")
	assert.NotEmpty(t, lines[1])
	assert.Equal(t, "Test summary content", lines[1])
}

// TestRollingSummary_EdgeCase_LLMFailure verifies that if summarization fails,
// the RollingSummary is NOT updated and the history remains unchanged.
func TestRollingSummary_EdgeCase_LLMFailure(t *testing.T) {
	tmpDir := t.TempDir()
	cfg, agent, cleanup := setupRollingSummaryTest(t, tmpDir)
	defer cleanup()

	agent.CompressionCfg.ChunkSizeTokens = 1
	// Fail on the second call: first call is main LLM, second is summarization
	provider := &mockSummarizationProvider{
		failOnCall: 2,
		failError:  fmt.Errorf("LLM unavailable"),
	}
	agent.Provider = provider

	sessionKey := "agent:main:llmfail"
	msgs := []testMessage{
		{Role: "user", Content: "Should not be summarized"},
		{Role: "assistant", Content: "Agreed"},
	}
	addTestMessages(agent, sessionKey, msgs...)

	al := createTestAgentLoop(t, cfg, agent)
	al.registry.agents["main"] = agent

	ctx := context.Background()
	_, err := al.processMessage(ctx, bus.InboundMessage{
		Channel:    "test",
		SenderID:   "user",
		ChatID:     "chat1",
		Content:    "Trigger",
		SessionKey: sessionKey,
	})
	// Summarization failure should be logged but NOT cause processMessage to return error
	require.NoError(t, err, "processMessage should succeed even if summarization fails")

	// RollingSummary should NOT have been created/updated
	summary := agent.Sessions.GetRollingSummary(sessionKey)
	assert.Empty(t, summary, "RollingSummary should remain empty after LLM failure")

	// History should be unchanged (not truncated)
	history := agent.Sessions.GetHistory(sessionKey)
	assert.GreaterOrEqual(t, len(history), 3, "history should not be truncated on LLM failure")
}

// TestRollingSummary_EdgeCase_ColdStorageNotConfigured verifies that if ColdStorage is nil
// (not configured), the RollingSummary is NOT updated and history is NOT truncated.
// This covers the scenario where archiving is not set up.
func TestRollingSummary_EdgeCase_ColdStorageNotConfigured(t *testing.T) {
	tmpDir := t.TempDir()
	cfg, agent, cleanup := setupRollingSummaryTest(t, tmpDir)
	defer cleanup()

	provider := &mockSummarizationProvider{summary: "Good summary"}
	agent.Provider = provider

	agent.CompressionCfg.ChunkSizeTokens = 1

	// Disable cold storage to simulate "not configured" scenario
	agent.ColdStorage = nil

	sessionKey := "agent:main:coldnil"
	msgs := []testMessage{
		{Role: "user", Content: "This should not be archived"},
		{Role: "assistant", Content: "Indeed"},
	}
	addTestMessages(agent, sessionKey, msgs...)

	al := createTestAgentLoop(t, cfg, agent)
	al.registry.agents["main"] = agent

	ctx := context.Background()
	al.processMessage(ctx, bus.InboundMessage{
		Channel:    "test",
		SenderID:   "user",
		ChatID:     "chat1",
		Content:    "Trigger",
		SessionKey: sessionKey,
	})

	// RollingSummary should NOT be updated because archiving not configured
	summary := agent.Sessions.GetRollingSummary(sessionKey)
	assert.Empty(t, summary, "RollingSummary should not be updated when cold storage is nil")

	// History should NOT be truncated
	history := agent.Sessions.GetHistory(sessionKey)
	assert.GreaterOrEqual(t, len(history), 3, "history should not be truncated when cold storage is nil")
}

// TestRollingSummary_Concurrency verifies that concurrent access to RollingSummary
// via SessionManager is safe with RWMutex.
func TestRollingSummary_Concurrency(t *testing.T) {
	tmpDir := t.TempDir()
	sm := session.NewSessionManager(filepath.Join(tmpDir, "sessions"))

	key := "agent:main:concurrent"
	sm.GetOrCreate(key)

	// Perform concurrent reads and writes
	done := make(chan bool, 50)
	for i := 0; i < 50; i++ {
		go func(idx int) {
			if idx%2 == 0 {
				sm.SetRollingSummary(key, fmt.Sprintf("Summary %d", idx))
			} else {
				_ = sm.GetRollingSummary(key)
			}
			done <- true
		}(i)
	}

	// Wait for all goroutines
	for i := 0; i < 50; i++ {
		<-done
	}

	// Should not have panicked and final value should be set
	final := sm.GetRollingSummary(key)
	assert.Contains(t, final, "Summary")
}

// ========== Test Helpers ==========

type testMessage struct {
	Role    string
	Content string
}

func addTestMessages(agent *AgentInstance, sessionKey string, msgs ...testMessage) {
	for _, tm := range msgs {
		agent.Sessions.AddMessage(sessionKey, tm.Role, tm.Content)
	}
}

func setupRollingSummaryTest(t *testing.T, tmpDir string) (*config.Config, *AgentInstance, func()) {
	workspace := filepath.Join(tmpDir, "workspace")
	require.NoError(t, os.MkdirAll(workspace, 0o755))

	cfg := &config.Config{
		Agents: config.AgentsConfig{
			Defaults: config.AgentDefaults{
				Workspace:         workspace,
				Model:             "test-model",
				MaxTokens:         4096,
				MaxToolIterations: 10,
			},
		},
		Compression: config.CompressionConfig{
			ChunkSizeTokens:   1, // very low to force compression in tests
			MinChunkMessages:  2,
			ContinuityBuffer:  1,
		},
	}

	agent := &AgentInstance{
		ID:             "default",
		Name:           "Test Agent",
		Model:          "test-model",
		Workspace:      workspace,
		MaxIterations:  10,
		MaxTokens:      4096,
		ContextWindow:  4096,
		Provider:       &mockSummarizationProvider{summary: "Default"},
		Sessions:       session.NewSessionManager(filepath.Join(workspace, "sessions")),
		ContextBuilder: NewContextBuilder(workspace),
		Tools:          tools.NewToolRegistry(),
		CompressionCfg: cfg.Compression,
	}

	cs, _ := NewColdStorage(filepath.Join(workspace, "cold_storage"))
	agent.ColdStorage = cs

	cleanup := func() {}

	return cfg, agent, cleanup
}

func createTestAgentLoop(t *testing.T, cfg *config.Config, agent *AgentInstance) *AgentLoop {
	msgBus := bus.NewMessageBus()
	provider := agent.Provider
	al := NewAgentLoop(cfg, msgBus, provider)
	return al
}

// createRecordingProvider wraps a provider and records the messages passed to Chat
func createRecordingProvider(delegate providers.LLMProvider, recordedMsgs *[]providers.Message) providers.LLMProvider {
	return &recordingProvider{
		delegate:     delegate,
		recordedMsgs: recordedMsgs,
	}
}

type recordingProvider struct {
	delegate     providers.LLMProvider
	recordedMsgs *[]providers.Message
}

func (rp *recordingProvider) Chat(
	ctx context.Context,
	messages []providers.Message,
	tools []providers.ToolDefinition,
	model string,
	opts map[string]any,
) (*providers.LLMResponse, error) {
	// Record the entire messages slice for verification
	*rp.recordedMsgs = messages
	return rp.delegate.Chat(ctx, messages, tools, model, opts)
}

func (rp *recordingProvider) GetDefaultModel() string {
	return rp.delegate.GetDefaultModel()
}
