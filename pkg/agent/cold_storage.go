package agent

import (
        "compress/gzip"
        "crypto/sha256"
        "encoding/json"
        "fmt"
        "os"
        "path/filepath"
        "strings"
        "sync"
        "time"

        "github.com/sipeed/picoclaw/pkg/logger"
        "github.com/sipeed/picoclaw/pkg/providers"
)

// ChunkRecord is the full archive record written to disk as <id>.json.gz
type ChunkRecord struct {
        ID         string              `json:"id"`
        SessionKey string              `json:"session_key"`
        MsgRange   [2]int              `json:"msg_range"`
        CreatedAt  time.Time           `json:"created_at"`
        Summary    string              `json:"summary"`
        Messages   []providers.Message `json:"messages"` // user + assistant only, no tool
}

// ChunkRef is the lightweight in-memory reference used for system prompt injection.
type ChunkRef struct {
        ID      string
        Summary string
}

// ColdStorage manages chunk archival and retrieval for all sessions.
type ColdStorage struct {
        dir      string
        counters map[string]int
        refs     map[string][]ChunkRef
        mu       sync.Mutex
}

// NewColdStorage creates (or opens) the storage directory and rebuilds the in-memory index.
func NewColdStorage(dir string) (*ColdStorage, error) {
        if err := os.MkdirAll(dir, 0o755); err != nil {
                return nil, fmt.Errorf("cold_storage: create dir %s: %w", dir, err)
        }
        cs := &ColdStorage{
                dir:      dir,
                counters: make(map[string]int),
                refs:     make(map[string][]ChunkRef),
        }
        if err := cs.RebuildIndex(); err != nil {
                logger.WarnCF("memory", "Cold storage index rebuild failed", map[string]any{"error": err.Error()})
        }
        return cs, nil
}

// RebuildIndex scans the storage directory on startup to seed counters and refs.
// Called once at startup; safe to call again to re-sync after external changes.
func (cs *ColdStorage) RebuildIndex() error {
        cs.mu.Lock()
        defer cs.mu.Unlock()

        entries, err := os.ReadDir(cs.dir)
        if err != nil {
                return err
        }

        cs.counters = make(map[string]int)
        cs.refs = make(map[string][]ChunkRef)

        for _, entry := range entries {
                if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json.gz") {
                        continue
                }

                record, err := cs.loadChunkLocked(strings.TrimSuffix(entry.Name(), ".json.gz"))
                if err != nil {
                        logger.WarnCF("memory", "Failed to load chunk during index rebuild",
                                map[string]any{"file": entry.Name(), "error": err.Error()})
                        continue
                }

                // Count chunks per session to seed the monotonic counter
                cs.counters[record.SessionKey]++

                // Add to ordered refs list
                cs.refs[record.SessionKey] = append(cs.refs[record.SessionKey], ChunkRef{
                        ID:      record.ID,
                        Summary: record.Summary,
                })
        }

        logger.InfoCF("memory", "[MEMORY] Index rebuilt",
                map[string]any{
                        "directory": cs.dir,
                        "sessions":  len(cs.counters),
                })
        return nil
}

// NextChunkID generates a unique, deterministic chunk ID for a session.
// ID = first 8 hex chars of SHA256(sessionKey + ":" + counter + ":" + nanoseconds)
func (cs *ColdStorage) NextChunkID(sessionKey string) string {
        cs.mu.Lock()
        cs.counters[sessionKey]++
        counter := cs.counters[sessionKey]
        cs.mu.Unlock()

        raw := fmt.Sprintf("%s:%d:%d", sessionKey, counter, time.Now().UnixNano())
        h := sha256.Sum256([]byte(raw))
        return fmt.Sprintf("%x", h[:4]) // 8 hex chars
}

// SaveChunk writes a ChunkRecord to disk atomically and updates the in-memory index.
func (cs *ColdStorage) SaveChunk(record ChunkRecord) error {
        data, err := json.Marshal(record)
        if err != nil {
                return fmt.Errorf("cold_storage: marshal: %w", err)
        }

        tmpFile, err := os.CreateTemp(cs.dir, "chunk-*.tmp")
        if err != nil {
                return fmt.Errorf("cold_storage: create temp: %w", err)
        }
        tmpPath := tmpFile.Name()

        cleanup := true
        defer func() {
                if cleanup {
                        _ = os.Remove(tmpPath)
                }
        }()

        gz := gzip.NewWriter(tmpFile)
        if _, err := gz.Write(data); err != nil {
                _ = tmpFile.Close()
                return fmt.Errorf("cold_storage: gzip write: %w", err)
        }
        if err := gz.Close(); err != nil {
                _ = tmpFile.Close()
                return fmt.Errorf("cold_storage: gzip close: %w", err)
        }
        if err := tmpFile.Sync(); err != nil {
                _ = tmpFile.Close()
                return err
        }
        if err := tmpFile.Close(); err != nil {
                return err
        }

        dest := filepath.Join(cs.dir, record.ID+".json.gz")
        if err := os.Rename(tmpPath, dest); err != nil {
                return fmt.Errorf("cold_storage: rename: %w", err)
        }
        cleanup = false

        // Update in-memory index
        cs.mu.Lock()
        cs.refs[record.SessionKey] = append(cs.refs[record.SessionKey], ChunkRef{
                ID:      record.ID,
                Summary: record.Summary,
        })
        cs.mu.Unlock()

        return nil
}

// LoadChunk reads a chunk from disk by ID.
func (cs *ColdStorage) LoadChunk(id string) (*ChunkRecord, error) {
        cs.mu.Lock()
        defer cs.mu.Unlock()
        return cs.loadChunkLocked(id)
}

func (cs *ColdStorage) loadChunkLocked(id string) (*ChunkRecord, error) {
        path := filepath.Join(cs.dir, id+".json.gz")
        f, err := os.Open(path)
        if err != nil {
                return nil, fmt.Errorf("cold_storage: open %s: %w", id, err)
        }
        defer f.Close()

        gz, err := gzip.NewReader(f)
        if err != nil {
                return nil, fmt.Errorf("cold_storage: gzip reader %s: %w", id, err)
        }
        defer gz.Close()

        var record ChunkRecord
        if err := json.NewDecoder(gz).Decode(&record); err != nil {
                return nil, fmt.Errorf("cold_storage: decode %s: %w", id, err)
        }
        return &record, nil
}

// ListRefs returns the ordered list of ChunkRefs for a session (for system prompt injection).
func (cs *ColdStorage) ListRefs(sessionKey string) []ChunkRef {
        cs.mu.Lock()
        defer cs.mu.Unlock()
        refs := cs.refs[sessionKey]
        if len(refs) == 0 {
                return nil
        }
        // Return a copy to avoid external mutation
        out := make([]ChunkRef, len(refs))
        copy(out, refs)
        return out
}
