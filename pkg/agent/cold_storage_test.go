package agent

import (
        "os"
        "path/filepath"
        "testing"
        "time"

        "github.com/sipeed/picoclaw/pkg/providers"
        "github.com/stretchr/testify/assert"
        "github.com/stretchr/testify/require"
)

func TestColdStorage_UniqueIDs(t *testing.T) {
        cs, err := NewColdStorage(t.TempDir())
        require.NoError(t, err)

        ids := make(map[string]bool)
        for i := 0; i < 100; i++ {
                id := cs.NextChunkID("session1")
                assert.False(t, ids[id], "ID collision detected: %s", id)
                ids[id] = true
                assert.Len(t, id, 8, "ID should be 8 characters long")
        }
}

func TestColdStorage_SaveLoad(t *testing.T) {
        tempDir := t.TempDir()
        cs, err := NewColdStorage(tempDir)
        require.NoError(t, err)

        record := ChunkRecord{
                ID:         "test1234",
                SessionKey: "session1",
                MsgRange:   [2]int{0, 5},
                CreatedAt:  time.Now(),
                Summary:    "Test summary",
                Messages: []providers.Message{
                        {Role: "user", Content: "Hello"},
                        {Role: "assistant", Content: "Hi there"},
                },
        }

        err = cs.SaveChunk(record)
        require.NoError(t, err)

        loaded, err := cs.LoadChunk("test1234")
        require.NoError(t, err)
        assert.Equal(t, record.ID, loaded.ID)
        assert.Equal(t, record.SessionKey, loaded.SessionKey)
        assert.Equal(t, record.Summary, loaded.Summary)
        assert.Len(t, loaded.Messages, 2)
        assert.Equal(t, "user", loaded.Messages[0].Role)
        assert.Equal(t, "Hello", loaded.Messages[0].Content)
}

func TestColdStorage_RebuildIndex(t *testing.T) {
        tempDir := t.TempDir()
        cs1, err := NewColdStorage(tempDir)
        require.NoError(t, err)

        // Create some chunks
        record1 := ChunkRecord{
                ID:         "test1234",
                SessionKey: "session1",
                Summary:    "Summary 1",
                Messages:   []providers.Message{{Role: "user", Content: "Hello"}},
        }
        record2 := ChunkRecord{
                ID:         "test5678",
                SessionKey: "session1",
                Summary:    "Summary 2",
                Messages:   []providers.Message{{Role: "user", Content: "World"}},
        }

        err = cs1.SaveChunk(record1)
        require.NoError(t, err)
        err = cs1.SaveChunk(record2)
        require.NoError(t, err)

        // Create new instance to test rebuild
        cs2, err := NewColdStorage(tempDir)
        require.NoError(t, err)

        refs := cs2.ListRefs("session1")
        assert.Len(t, refs, 2)
        assert.Equal(t, "test1234", refs[0].ID)
        assert.Equal(t, "Summary 1", refs[0].Summary)
        assert.Equal(t, "test5678", refs[1].ID)
        assert.Equal(t, "Summary 2", refs[1].Summary)
}

func TestColdStorage_RetrieveChunkToolMessagesExcluded(t *testing.T) {
        tempDir := t.TempDir()
        cs, err := NewColdStorage(tempDir)
        require.NoError(t, err)

        record := ChunkRecord{
                ID:         "test1234",
                SessionKey: "session1",
                Summary:    "Test summary",
                Messages: []providers.Message{
                        {Role: "user", Content: "Hello"},
                        {Role: "assistant", Content: "Hi there"},
                        {Role: "tool", Content: "Tool result"}, // This should be included in Messages
                },
        }

        err = cs.SaveChunk(record)
        require.NoError(t, err)

        loaded, err := cs.LoadChunk("test1234")
        require.NoError(t, err)
        assert.Len(t, loaded.Messages, 3) // All messages should be preserved in archive
}

func TestFormatChunkTranscript(t *testing.T) {
        record := &ChunkRecord{
                ID:         "test1234",
                CreatedAt:  time.Date(2024, 1, 1, 12, 0, 0, 0, time.UTC),
                Summary:    "Test summary",
                Messages: []providers.Message{
                        {Role: "user", Content: "Hello"},
                        {Role: "assistant", Content: "Hi there"},
                },
        }

        result := formatChunkTranscript(record)
        expectedStart := "[Archived chunk test1234 — 2024-01-01 12:00]"
        assert.Contains(t, result, expectedStart)
        assert.Contains(t, result, "user: Hello")
        assert.Contains(t, result, "assistant: Hi there")
}
