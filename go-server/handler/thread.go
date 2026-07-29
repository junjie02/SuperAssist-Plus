package handler

import (
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"gorm.io/gorm"

	"superassist-go/model"
)

type ThreadHandler struct {
	db        *gorm.DB
	dataDir   string
	historyMu sync.Mutex
}

type ThreadSummary struct {
	ThreadID     string `json:"thread_id"`
	UpdatedAt    string `json:"updated_at"`
	Preview      string `json:"preview"`
	MessageCount int    `json:"message_count"`
}

type HistoryItem struct {
	RecordIndex int    `json:"record_index"`
	Role        string `json:"role"`
	Content     string `json:"content"`
	CreatedAt   string `json:"created_at"`
}

type AdminUserSummary struct {
	ID                string `json:"id"`
	Username          string `json:"username"`
	Channel           string `json:"channel"`
	IsAdmin           bool   `json:"is_admin"`
	CreatedAt         string `json:"created_at,omitempty"`
	LastActiveAt      string `json:"last_active_at,omitempty"`
	ConversationCount int    `json:"conversation_count"`
	MessageCount      int    `json:"message_count"`
}

type threadMetadata struct {
	UserID string `json:"user_id"`
}

type channelThreadEntry struct {
	ThreadID string `json:"thread_id"`
	UserID   string `json:"user_id"`
}

func NewThreadHandler(db *gorm.DB, dataDir string) *ThreadHandler {
	return &ThreadHandler{db: db, dataDir: dataDir}
}

// GetThreads lists threads owned by the current user.
func (h *ThreadHandler) GetThreads(c *gin.Context) {
	c.JSON(http.StatusOK, h.listThreads(c.GetString("user_id")))
}

// GetHistory returns a current user's thread history.
func (h *ThreadHandler) GetHistory(c *gin.Context) {
	threadID := c.Param("id")
	if !h.ownsThread(c.GetString("user_id"), threadID) {
		c.JSON(http.StatusNotFound, gin.H{"detail": "thread not found"})
		return
	}
	h.writeHistory(c, threadID)
}

// DeleteThread removes a thread only when it belongs to the current user.
func (h *ThreadHandler) DeleteThread(c *gin.Context) {
	threadID := c.Param("id")
	if !h.ownsThread(c.GetString("user_id"), threadID) {
		c.JSON(http.StatusNotFound, gin.H{"detail": "thread not found"})
		return
	}
	threadDir, ok := h.threadDir(threadID)
	if !ok {
		c.JSON(http.StatusBadRequest, gin.H{"detail": "invalid thread id"})
		return
	}
	if err := os.RemoveAll(threadDir); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": "failed to delete thread"})
		return
	}
	c.JSON(http.StatusOK, gin.H{"deleted": true})
}

// GetAdminUsers lists registered and channel-backed identities with usage totals.
func (h *ThreadHandler) GetAdminUsers(c *gin.Context) {
	owners := h.ownerIndex()
	users := make(map[string]*AdminUserSummary)
	var registered []model.User
	_ = h.db.Order("created_at ASC").Find(&registered).Error
	for _, user := range registered {
		users[user.ID] = &AdminUserSummary{
			ID:        user.ID,
			Username:  user.Username,
			Channel:   "web",
			IsAdmin:   user.IsAdmin,
			CreatedAt: user.CreatedAt.UTC().Format(time.RFC3339),
		}
	}
	for _, userID := range owners {
		if _, exists := users[userID]; !exists {
			users[userID] = &AdminUserSummary{
				ID:       userID,
				Username: identityLabel(userID),
				Channel:  identityChannel(userID),
			}
		}
	}
	for userID, user := range users {
		threads := h.listThreadsWithOwners(userID, owners)
		user.ConversationCount = len(threads)
		for _, thread := range threads {
			user.MessageCount += thread.MessageCount
			if user.LastActiveAt == "" || thread.UpdatedAt > user.LastActiveAt {
				user.LastActiveAt = thread.UpdatedAt
			}
		}
	}
	result := make([]AdminUserSummary, 0, len(users))
	for _, user := range users {
		result = append(result, *user)
	}
	sort.Slice(result, func(i, j int) bool {
		if result[i].IsAdmin != result[j].IsAdmin {
			return result[i].IsAdmin
		}
		if result[i].LastActiveAt != result[j].LastActiveAt {
			return result[i].LastActiveAt > result[j].LastActiveAt
		}
		return strings.ToLower(result[i].Username) < strings.ToLower(result[j].Username)
	})
	c.JSON(http.StatusOK, result)
}

func (h *ThreadHandler) GetAdminUserThreads(c *gin.Context) {
	userID := c.Param("user_id")
	owners := h.ownerIndex()
	if !h.identityExists(userID, owners) {
		c.JSON(http.StatusNotFound, gin.H{"detail": "user not found"})
		return
	}
	c.JSON(http.StatusOK, h.listThreadsWithOwners(userID, owners))
}

func (h *ThreadHandler) GetAdminUserHistory(c *gin.Context) {
	userID := c.Param("user_id")
	threadID := c.Param("thread_id")
	if h.ownerIndex()[threadID] != userID {
		c.JSON(http.StatusNotFound, gin.H{"detail": "thread not found for user"})
		return
	}
	h.writeHistory(c, threadID)
}

// DeleteAdminUserMessage removes one persisted short-memory record.
func (h *ThreadHandler) DeleteAdminUserMessage(c *gin.Context) {
	userID := c.Param("user_id")
	threadID := c.Param("thread_id")
	if h.ownerIndex()[threadID] != userID {
		c.JSON(http.StatusNotFound, gin.H{"detail": "thread not found for user"})
		return
	}
	recordIndex, err := strconv.Atoi(c.Param("record_index"))
	if err != nil || recordIndex < 0 {
		c.JSON(http.StatusBadRequest, gin.H{"detail": "invalid message index"})
		return
	}
	threadDir, ok := h.threadDir(threadID)
	if !ok {
		c.JSON(http.StatusBadRequest, gin.H{"detail": "invalid thread id"})
		return
	}

	h.historyMu.Lock()
	defer h.historyMu.Unlock()
	remaining, err := deleteHistoryRecord(filepath.Join(threadDir, "messages.jsonl"), recordIndex)
	if os.IsNotExist(err) || err == errHistoryRecordNotFound {
		c.JSON(http.StatusNotFound, gin.H{"detail": "message not found"})
		return
	}
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": "failed to delete message"})
		return
	}
	c.JSON(http.StatusOK, gin.H{"deleted": true, "remaining": remaining})
}

// CanAccessThread protects the WebSocket path before a supplied thread is sent to Python.
func (h *ThreadHandler) CanAccessThread(userID, threadID string) bool {
	if threadID == "" {
		return true
	}
	threadDir, ok := h.threadDir(threadID)
	if !ok {
		return false
	}
	if _, err := os.Stat(threadDir); os.IsNotExist(err) {
		return true
	}
	return h.ownerIndex()[threadID] == userID
}

func (h *ThreadHandler) listThreads(userID string) []ThreadSummary {
	return h.listThreadsWithOwners(userID, h.ownerIndex())
}

func (h *ThreadHandler) listThreadsWithOwners(userID string, owners map[string]string) []ThreadSummary {
	entries, err := os.ReadDir(filepath.Join(h.dataDir, "threads"))
	if err != nil {
		return []ThreadSummary{}
	}
	summaries := make([]ThreadSummary, 0)
	for _, entry := range entries {
		if !entry.IsDir() || owners[entry.Name()] != userID {
			continue
		}
		messagesPath := filepath.Join(h.dataDir, "threads", entry.Name(), "messages.jsonl")
		info, err := os.Stat(messagesPath)
		if err != nil {
			continue
		}
		history := readHistory(messagesPath)
		preview := firstUserPreview(history)
		if preview == "" {
			preview = "(empty thread)"
		}
		summaries = append(summaries, ThreadSummary{
			ThreadID:     entry.Name(),
			UpdatedAt:    info.ModTime().UTC().Format(time.RFC3339),
			Preview:      preview,
			MessageCount: len(history),
		})
	}
	sort.Slice(summaries, func(i, j int) bool { return summaries[i].UpdatedAt > summaries[j].UpdatedAt })
	return summaries
}

func (h *ThreadHandler) writeHistory(c *gin.Context, threadID string) {
	threadDir, ok := h.threadDir(threadID)
	if !ok {
		c.JSON(http.StatusBadRequest, gin.H{"detail": "invalid thread id"})
		return
	}
	path := filepath.Join(threadDir, "messages.jsonl")
	if _, err := os.Stat(path); err != nil {
		c.JSON(http.StatusNotFound, gin.H{"detail": "thread not found"})
		return
	}
	c.JSON(http.StatusOK, readHistory(path))
}

func (h *ThreadHandler) ownsThread(userID, threadID string) bool {
	if _, ok := h.threadDir(threadID); !ok {
		return false
	}
	return h.ownerIndex()[threadID] == userID
}

func (h *ThreadHandler) ownerIndex() map[string]string {
	owners := make(map[string]string)
	threadsDir := filepath.Join(h.dataDir, "threads")
	entries, _ := os.ReadDir(threadsDir)
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		data, err := os.ReadFile(filepath.Join(threadsDir, entry.Name(), "thread_meta.json"))
		if err != nil {
			continue
		}
		var metadata threadMetadata
		if json.Unmarshal(data, &metadata) == nil && metadata.UserID != "" {
			owners[entry.Name()] = metadata.UserID
		}
	}
	for _, filename := range []string{"feishu_threads.json", "wecom_threads.json"} {
		data, err := os.ReadFile(filepath.Join(h.dataDir, "channels", filename))
		if err != nil {
			continue
		}
		var channelEntries map[string]channelThreadEntry
		if json.Unmarshal(data, &channelEntries) != nil {
			continue
		}
		for _, item := range channelEntries {
			if item.ThreadID != "" && item.UserID != "" {
				owners[item.ThreadID] = item.UserID
			}
		}
	}

	var registered []model.User
	if h.db.Select("id").Find(&registered).Error == nil && len(registered) == 1 {
		for _, entry := range entries {
			if entry.IsDir() {
				if _, found := owners[entry.Name()]; !found {
					owners[entry.Name()] = registered[0].ID
				}
			}
		}
	}
	return owners
}

func (h *ThreadHandler) identityExists(userID string, owners map[string]string) bool {
	var count int64
	if h.db.Model(&model.User{}).Where("id = ?", userID).Count(&count).Error == nil && count > 0 {
		return true
	}
	for _, ownerID := range owners {
		if ownerID == userID {
			return true
		}
	}
	return false
}

func (h *ThreadHandler) threadDir(threadID string) (string, bool) {
	if threadID == "" || filepath.Base(threadID) != threadID || strings.ContainsAny(threadID, `/\\`) {
		return "", false
	}
	return filepath.Join(h.dataDir, "threads", threadID), true
}

func readHistory(path string) []HistoryItem {
	records, _ := readHistoryRecords(path)
	items := make([]HistoryItem, 0)
	for index, record := range records {
		raw := record.value
		role, _ := raw["role"].(string)
		content, _ := raw["content"].(string)
		createdAt, _ := raw["created_at"].(string)
		items = append(items, HistoryItem{RecordIndex: index, Role: role, Content: content, CreatedAt: createdAt})
	}
	return items
}

var errHistoryRecordNotFound = errors.New("history record not found")

type historyRecord struct {
	raw   string
	value map[string]any
}

func readHistoryRecords(path string) ([]historyRecord, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	records := make([]historyRecord, 0)
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var value map[string]any
		if json.Unmarshal([]byte(line), &value) != nil {
			continue
		}
		records = append(records, historyRecord{raw: line, value: value})
	}
	return records, nil
}

func deleteHistoryRecord(path string, recordIndex int) (int, error) {
	records, err := readHistoryRecords(path)
	if err != nil {
		return 0, err
	}
	if recordIndex >= len(records) {
		return len(records), errHistoryRecordNotFound
	}
	records = append(records[:recordIndex], records[recordIndex+1:]...)
	var content strings.Builder
	for _, record := range records {
		content.WriteString(record.raw)
		content.WriteByte('\n')
	}
	if err := os.WriteFile(path, []byte(content.String()), 0o600); err != nil {
		return 0, err
	}
	return len(records), nil
}

func firstUserPreview(history []HistoryItem) string {
	for _, item := range history {
		if item.Role != "user" {
			continue
		}
		content := []rune(item.Content)
		if len(content) > 120 {
			return string(content[:120]) + "..."
		}
		return item.Content
	}
	return ""
}

func identityChannel(userID string) string {
	switch {
	case strings.HasPrefix(userID, "feishu-group:"):
		return "feishu-group"
	case strings.HasPrefix(userID, "feishu:"):
		return "feishu"
	case strings.HasPrefix(userID, "wecom-rpa-group:"):
		return "wecom-rpa"
	case strings.HasPrefix(userID, "wecom-group:"):
		return "wecom-group"
	case strings.HasPrefix(userID, "wecom:"):
		return "wecom"
	default:
		return "external"
	}
}

func identityLabel(userID string) string {
	channel := identityChannel(userID)
	suffix := userID
	if len([]rune(suffix)) > 12 {
		runes := []rune(suffix)
		suffix = string(runes[len(runes)-12:])
	}
	switch channel {
	case "feishu-group":
		return "Feishu group · " + suffix
	case "feishu":
		return "Feishu user · " + suffix
	case "wecom-rpa", "wecom-group":
		return "WeCom group · " + suffix
	case "wecom":
		return "WeCom user · " + suffix
	default:
		return userID
	}
}
