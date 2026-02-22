package tools

import (
        "context"
        "fmt"
        "strings"
)

// retrieveChunkFunc is injected at registration time to avoid circular imports.
// Signature: func(chunkID string) (transcript string, err error)
type retrieveChunkFunc func(chunkID string) (string, error)

// RetrieveChunkTool allows the agent to load a full archived conversation chunk by ID.
type RetrieveChunkTool struct {
        retrieveFn retrieveChunkFunc
}

// NewRetrieveChunkTool creates the tool with an injected retrieval function.
func NewRetrieveChunkTool(fn retrieveChunkFunc) *RetrieveChunkTool {
        return &RetrieveChunkTool{retrieveFn: fn}
}

func (t *RetrieveChunkTool) Name() string { return "retrieve_chunk" }

func (t *RetrieveChunkTool) Description() string {
        return "Load the full archived messages for a past conversation chunk by its ID. " +
                "Use chunk IDs listed in the [Memory] section of the system prompt."
}

func (t *RetrieveChunkTool) Parameters() map[string]any {
        return map[string]any{
                "type": "object",
                "properties": map[string]any{
                        "chunk_id": map[string]any{
                                "type":        "string",
                                "description": "The 8-character chunk ID (e.g. a3f72b1c)",
                        },
                },
                "required": []string{"chunk_id"},
        }
}

func (t *RetrieveChunkTool) Execute(_ context.Context, args map[string]any) *ToolResult {
        chunkID, _ := args["chunk_id"].(string)
        chunkID = strings.TrimSpace(chunkID)
        if chunkID == "" {
                return ErrorResult("chunk_id is required").WithError(fmt.Errorf("missing chunk_id"))
        }

        transcript, err := t.retrieveFn(chunkID)
        if err != nil {
                return ErrorResult(fmt.Sprintf("chunk %s not found: %v", chunkID, err)).WithError(err)
        }

        return &ToolResult{
                ForLLM: transcript,
                // Silent=false so the agent sees the result.
                // Role=tool is assigned by the caller (loop.go).
        }
}
