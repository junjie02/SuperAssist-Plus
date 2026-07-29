package handler

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/glebarez/sqlite"
	"gorm.io/gorm"

	"superassist-go/model"
)

func TestThreadHandlerEnforcesOwnershipAndListsChannelUsers(t *testing.T) {
	gin.SetMode(gin.TestMode)
	db, err := gorm.Open(sqlite.Open(":memory:"), &gorm.Config{})
	if err != nil {
		t.Fatal(err)
	}
	if err := db.AutoMigrate(&model.User{}); err != nil {
		t.Fatal(err)
	}
	painting := model.User{ID: "user_painting", Username: "painting", PasswordHash: "x", IsAdmin: true}
	alice := model.User{ID: "user_alice", Username: "alice", PasswordHash: "x"}
	if err := db.Create(&painting).Error; err != nil {
		t.Fatal(err)
	}
	if err := db.Create(&alice).Error; err != nil {
		t.Fatal(err)
	}

	dataDir := t.TempDir()
	writeTestThread(t, dataDir, "thread_p", painting.ID, "painting question")
	writeTestThread(t, dataDir, "thread_a", alice.ID, "alice question")
	writeTestThread(t, dataDir, "feishu_1", "", "feishu question")
	channelDir := filepath.Join(dataDir, "channels")
	if err := os.MkdirAll(channelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	mapping := map[string]channelThreadEntry{
		"feishu:chat:topic": {ThreadID: "feishu_1", UserID: "feishu:ou_1"},
	}
	mappingData, _ := json.Marshal(mapping)
	if err := os.WriteFile(filepath.Join(channelDir, "feishu_threads.json"), mappingData, 0o600); err != nil {
		t.Fatal(err)
	}

	handler := NewThreadHandler(db, dataDir)

	listResponse := invokeHandler(t, handler.GetThreads, painting.ID, nil)
	if listResponse.Code != http.StatusOK {
		t.Fatalf("list status = %d", listResponse.Code)
	}
	var threads []ThreadSummary
	if err := json.Unmarshal(listResponse.Body.Bytes(), &threads); err != nil {
		t.Fatal(err)
	}
	if len(threads) != 1 || threads[0].ThreadID != "thread_p" {
		t.Fatalf("painting threads = %#v", threads)
	}

	denied := invokeHandler(t, handler.GetHistory, painting.ID, gin.Params{{Key: "id", Value: "thread_a"}})
	if denied.Code != http.StatusNotFound {
		t.Fatalf("cross-user history status = %d", denied.Code)
	}
	if handler.CanAccessThread(painting.ID, "thread_a") {
		t.Fatal("cross-user websocket thread access was allowed")
	}

	usersResponse := invokeHandler(t, handler.GetAdminUsers, painting.ID, nil)
	var users []AdminUserSummary
	if err := json.Unmarshal(usersResponse.Body.Bytes(), &users); err != nil {
		t.Fatal(err)
	}
	if len(users) != 3 {
		t.Fatalf("admin users = %#v", users)
	}
	foundFeishu := false
	for _, user := range users {
		if user.ID == "feishu:ou_1" {
			foundFeishu = user.Channel == "feishu" && user.ConversationCount == 1 && user.MessageCount == 2
		}
	}
	if !foundFeishu {
		t.Fatalf("missing Feishu identity in %#v", users)
	}

	deleted := invokeHandler(t, handler.DeleteAdminUserMessage, painting.ID, gin.Params{
		{Key: "user_id", Value: alice.ID},
		{Key: "thread_id", Value: "thread_a"},
		{Key: "record_index", Value: "0"},
	})
	if deleted.Code != http.StatusOK {
		t.Fatalf("delete message status = %d: %s", deleted.Code, deleted.Body.String())
	}
	remainingHistory := invokeHandler(t, handler.GetAdminUserHistory, painting.ID, gin.Params{
		{Key: "user_id", Value: alice.ID},
		{Key: "thread_id", Value: "thread_a"},
	})
	var remaining []HistoryItem
	if err := json.Unmarshal(remainingHistory.Body.Bytes(), &remaining); err != nil {
		t.Fatal(err)
	}
	if len(remaining) != 1 || remaining[0].Role != "assistant" || remaining[0].RecordIndex != 0 {
		t.Fatalf("remaining history = %#v", remaining)
	}
}

func writeTestThread(t *testing.T, dataDir, threadID, userID, question string) {
	t.Helper()
	dir := filepath.Join(dataDir, "threads", threadID)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	meta, _ := json.Marshal(threadMetadata{UserID: userID})
	if err := os.WriteFile(filepath.Join(dir, "thread_meta.json"), meta, 0o600); err != nil {
		t.Fatal(err)
	}
	messages := `{"role":"user","content":"` + question + `","created_at":"2026-01-01T00:00:00Z"}` + "\n" +
		`{"role":"assistant","content":"answer","created_at":"2026-01-01T00:00:01Z"}` + "\n"
	if err := os.WriteFile(filepath.Join(dir, "messages.jsonl"), []byte(messages), 0o600); err != nil {
		t.Fatal(err)
	}
}

func invokeHandler(t *testing.T, handler gin.HandlerFunc, userID string, params gin.Params) *httptest.ResponseRecorder {
	t.Helper()
	recorder := httptest.NewRecorder()
	context, _ := gin.CreateTestContext(recorder)
	context.Request = httptest.NewRequest(http.MethodGet, "/", nil)
	context.Params = params
	context.Set("user_id", userID)
	handler(context)
	return recorder
}
