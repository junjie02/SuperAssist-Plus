package handler

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"gorm.io/gorm"
)

type ThreadHandler struct {
	db      *gorm.DB
	dataDir string
}

// ThreadSummary sent to the client.
type ThreadSummary struct {
	ThreadID  string `json:"thread_id"`
	UpdatedAt string `json:"updated_at"`
	Preview   string `json:"preview"`
}

// HistoryItem is one message in a thread.
type HistoryItem struct {
	Role      string `json:"role"`
	Content   string `json:"content"`
	CreatedAt string `json:"created_at"`
}

func NewThreadHandler(db *gorm.DB, dataDir string) *ThreadHandler {
	return &ThreadHandler{db: db, dataDir: dataDir}
}

// GetThreads lists all threads for the current user.
// GET /api/threads
func (h *ThreadHandler) GetThreads(c *gin.Context) {
	userID := c.GetString("user_id")

	threadsDir := filepath.Join(h.dataDir, "threads")
	entries, err := os.ReadDir(threadsDir)
	if err != nil {
		c.JSON(http.StatusOK, []ThreadSummary{})
		return
	}

	var summaries []ThreadSummary
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		threadID := entry.Name()
		messagesPath := filepath.Join(threadsDir, threadID, "messages.jsonl")

		info, statErr := os.Stat(messagesPath)
		if statErr != nil {
			continue
		}

		preview := h.readPreview(messagesPath, userID)
		if preview == "" {
			preview = "(empty thread)"
		}

		summaries = append(summaries, ThreadSummary{
			ThreadID:  threadID,
			UpdatedAt: info.ModTime().UTC().Format(time.RFC3339),
			Preview:   preview,
		})
	}

	// Sort newest first
	sort.Slice(summaries, func(i, j int) bool {
		return summaries[i].UpdatedAt > summaries[j].UpdatedAt
	})

	c.JSON(http.StatusOK, summaries)
}

// GetHistory returns the message history for a thread.
// GET /api/threads/:id/history
func (h *ThreadHandler) GetHistory(c *gin.Context) {
	threadID := c.Param("id")
	messagesPath := filepath.Join(h.dataDir, "threads", threadID, "messages.jsonl")

	data, err := os.ReadFile(messagesPath)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"detail": "thread not found"})
		return
	}

	var items []HistoryItem
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		var raw map[string]any
		if err := json.Unmarshal([]byte(line), &raw); err != nil {
			continue
		}

		role, _ := raw["role"].(string)
		content, _ := raw["content"].(string)
		createdAt, _ := raw["created_at"].(string)

		items = append(items, HistoryItem{
			Role:      role,
			Content:   content,
			CreatedAt: createdAt,
		})
	}

	c.JSON(http.StatusOK, items)
}

// DeleteThread removes a thread directory.
// DELETE /api/threads/:id
func (h *ThreadHandler) DeleteThread(c *gin.Context) {
	threadID := c.Param("id")
	threadDir := filepath.Join(h.dataDir, "threads", threadID)

	if err := os.RemoveAll(threadDir); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": "failed to delete thread"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"deleted": true})
}

func (h *ThreadHandler) readPreview(path, _userID string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}

	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	// Take the first user message as preview
	for _, line := range lines {
		var raw map[string]any
		if err := json.Unmarshal([]byte(line), &raw); err != nil {
			continue
		}
		role, _ := raw["role"].(string)
		if role == "user" {
			content, _ := raw["content"].(string)
			if len(content) > 120 {
				content = content[:120] + "..."
			}
			return content
		}
	}
	return ""
}
