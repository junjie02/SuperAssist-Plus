package proxy

import (
	"bytes"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestChatSendsRAGMode(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatalf("decode chat request: %v", err)
		}
		if !request.RAGMode {
			t.Fatal("rag_mode = false, want true")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"type\":\"done\"}\n\n")
	}))
	defer server.Close()

	client := NewPythonClient(server.URL)
	response, err := client.Chat("user-a", "thread-a", "question", true)
	if err != nil {
		t.Fatalf("Chat returned an error: %v", err)
	}
	response.Body.Close()
}

func TestUploadRAGDocumentsPreservesMultipartAndUser(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("user_id") != "user + one" {
			t.Fatalf("user_id = %q", r.URL.Query().Get("user_id"))
		}
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatalf("parse multipart: %v", err)
		}
		file, header, err := r.FormFile("files")
		if err != nil {
			t.Fatalf("read uploaded file: %v", err)
		}
		defer file.Close()
		if header.Filename != "notes.txt" {
			t.Fatalf("filename = %q", header.Filename)
		}
		content, _ := io.ReadAll(file)
		if string(content) != "hello" {
			t.Fatalf("content = %q", content)
		}
		w.WriteHeader(http.StatusAccepted)
		_, _ = io.WriteString(w, `{"documents":[]}`)
	}))
	defer server.Close()

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("files", "notes.txt")
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.WriteString(part, "hello")
	_ = writer.Close()

	client := NewPythonClient(server.URL)
	_, status, err := client.UploadRAGDocuments("user + one", writer.FormDataContentType(), &body)
	if err != nil {
		t.Fatalf("UploadRAGDocuments returned an error: %v", err)
	}
	if status != http.StatusAccepted {
		t.Fatalf("status = %d, want %d", status, http.StatusAccepted)
	}
}

func TestRAGGraphPreservesUser(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/internal/rag/graph" {
			t.Fatalf("path = %q", r.URL.Path)
		}
		if r.URL.Query().Get("user_id") != "user + one" {
			t.Fatalf("user_id = %q", r.URL.Query().Get("user_id"))
		}
		_, _ = io.WriteString(w, `{"nodes":[],"edges":[],"stats":{"nodes":0,"edges":0,"documents":0}}`)
	}))
	defer server.Close()

	client := NewPythonClient(server.URL)
	raw, status, err := client.RAGGraph("user + one")
	if err != nil {
		t.Fatalf("RAGGraph returned an error: %v", err)
	}
	if status != http.StatusOK || !json.Valid(raw) {
		t.Fatalf("status = %d, raw = %s", status, raw)
	}
}

func TestReadSSEEventsPreservesExtendedFields(t *testing.T) {
	body := io.NopCloser(strings.NewReader(
		"data: {\"type\":\"done\",\"thread_id\":\"thread_123\",\"answer\":\"hello\"}\n\n",
	))
	events := make(chan SSEEvent, 1)

	if err := ReadSSEEvents(body, events); err != nil {
		t.Fatalf("ReadSSEEvents returned an error: %v", err)
	}
	event := <-events

	if event.Type != "done" {
		t.Fatalf("event type = %q, want done", event.Type)
	}
	var raw map[string]any
	if err := json.Unmarshal(event.Raw, &raw); err != nil {
		t.Fatalf("raw event is not valid JSON: %v", err)
	}
	if raw["thread_id"] != "thread_123" {
		t.Fatalf("thread_id = %v, want thread_123", raw["thread_id"])
	}
	if raw["answer"] != "hello" {
		t.Fatalf("answer = %v, want hello", raw["answer"])
	}
}
