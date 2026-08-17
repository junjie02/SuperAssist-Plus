package proxy

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// PythonClient calls the Python AI engine internal endpoints.
type PythonClient struct {
	baseURL string
	client  *http.Client
}

func NewPythonClient(baseURL string) *PythonClient {
	return &PythonClient{
		baseURL: baseURL,
		client: &http.Client{
			Timeout: 300 * time.Second, // LLM calls can be slow
		},
	}
}

// ChatRequest is the payload sent to POST /internal/chat.
type ChatRequest struct {
	UserID   string `json:"user_id"`
	Message  string `json:"message"`
	ThreadID string `json:"thread_id,omitempty"`
	RAGMode  bool   `json:"rag_mode"`
}

// Chat sends a message to the Python AI engine and returns the SSE response body.
// The caller is responsible for closing the body.
func (p *PythonClient) Chat(userID, threadID, message string, ragMode bool) (*http.Response, error) {
	reqBody := ChatRequest{
		UserID:   userID,
		Message:  message,
		ThreadID: threadID,
		RAGMode:  ragMode,
	}
	data, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal chat request: %w", err)
	}

	resp, err := p.client.Post(
		p.baseURL+"/internal/chat",
		"application/json",
		bytes.NewReader(data),
	)
	if err != nil {
		return nil, fmt.Errorf("python chat request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("python chat error: status=%d body=%s", resp.StatusCode, string(body))
	}

	return resp, nil
}

// RAGDocuments returns the current user's uploaded-document manifest.
func (p *PythonClient) RAGDocuments(userID string) (json.RawMessage, int, error) {
	endpoint := fmt.Sprintf("%s/internal/rag/documents?user_id=%s", p.baseURL, url.QueryEscape(userID))
	return p.doRAGRequest(http.MethodGet, endpoint, "", nil)
}

// RAGGraph returns the current user's uploaded-document knowledge graph.
func (p *PythonClient) RAGGraph(userID string) (json.RawMessage, int, error) {
	endpoint := fmt.Sprintf("%s/internal/rag/graph?user_id=%s", p.baseURL, url.QueryEscape(userID))
	return p.doRAGRequest(http.MethodGet, endpoint, "", nil)
}

// UploadRAGDocuments streams a browser multipart request to the Python engine.
func (p *PythonClient) UploadRAGDocuments(
	userID string,
	contentType string,
	body io.Reader,
) (json.RawMessage, int, error) {
	endpoint := fmt.Sprintf("%s/internal/rag/documents?user_id=%s", p.baseURL, url.QueryEscape(userID))
	return p.doRAGRequest(http.MethodPost, endpoint, contentType, body)
}

// DeleteRAGDocument removes one document from the current user's hybrid RAG index.
func (p *PythonClient) DeleteRAGDocument(userID, documentID string) (json.RawMessage, int, error) {
	endpoint := fmt.Sprintf(
		"%s/internal/rag/documents/%s?user_id=%s",
		p.baseURL,
		url.PathEscape(documentID),
		url.QueryEscape(userID),
	)
	return p.doRAGRequest(http.MethodDelete, endpoint, "", nil)
}

func (p *PythonClient) doRAGRequest(
	method string,
	endpoint string,
	contentType string,
	body io.Reader,
) (json.RawMessage, int, error) {
	req, err := http.NewRequest(method, endpoint, body)
	if err != nil {
		return nil, 0, fmt.Errorf("create RAG request: %w", err)
	}
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	resp, err := p.client.Do(req)
	if err != nil {
		return nil, 0, fmt.Errorf("RAG request: %w", err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, 0, fmt.Errorf("read RAG response: %w", err)
	}
	if !json.Valid(data) {
		return nil, 0, fmt.Errorf("invalid RAG response: status=%d", resp.StatusCode)
	}
	return json.RawMessage(data), resp.StatusCode, nil
}

// SSEEvent represents a single SSE data line.
type SSEEvent struct {
	Type    string          `json:"type"`
	Content string          `json:"content,omitempty"`
	Message string          `json:"message,omitempty"`
	Raw     json.RawMessage `json:"-"`
}

// ReadSSEEvents reads SSE events from a response body and sends parsed
// JSON to the provided channel.  Returns when the stream ends.
func ReadSSEEvents(body io.ReadCloser, ch chan<- SSEEvent) error {
	defer close(ch)
	defer body.Close()

	scanner := bufio.NewScanner(body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := line[6:]
		var evt SSEEvent
		if err := json.Unmarshal([]byte(data), &evt); err != nil {
			// If unmarshal into SSEEvent fails, send raw
			evt = SSEEvent{Raw: json.RawMessage(data)}
		} else {
			// Preserve fields outside SSEEvent (for example done.answer and done.thread_id).
			evt.Raw = json.RawMessage(data)
		}
		ch <- evt
	}
	return scanner.Err()
}

// GraphPayload fetches the memory graph from Python.
func (p *PythonClient) GraphPayload(userID string) (json.RawMessage, error) {
	endpoint := fmt.Sprintf("%s/internal/graph?user_id=%s", p.baseURL, url.QueryEscape(userID))
	resp, err := p.client.Get(endpoint)
	if err != nil {
		return nil, fmt.Errorf("graph request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("graph error: status=%d", resp.StatusCode)
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read graph response: %w", err)
	}
	return json.RawMessage(data), nil
}

// SettingsPayload fetches the configurable memory and Feishu settings.
func (p *PythonClient) SettingsPayload() (json.RawMessage, error) {
	resp, err := p.client.Get(p.baseURL + "/internal/settings")
	if err != nil {
		return nil, fmt.Errorf("settings request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("settings error: status=%d", resp.StatusCode)
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read settings response: %w", err)
	}
	return json.RawMessage(data), nil
}

// UpdateSettings validates and persists settings through the Python engine.
func (p *PythonClient) UpdateSettings(payload []byte) (json.RawMessage, int, error) {
	req, err := http.NewRequest(http.MethodPut, p.baseURL+"/internal/settings", bytes.NewReader(payload))
	if err != nil {
		return nil, 0, fmt.Errorf("create settings request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := p.client.Do(req)
	if err != nil {
		return nil, 0, fmt.Errorf("settings request: %w", err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, 0, fmt.Errorf("read settings response: %w", err)
	}
	return json.RawMessage(data), resp.StatusCode, nil
}

// Health checks if the Python AI engine is reachable.
func (p *PythonClient) Health() error {
	resp, err := p.client.Get(p.baseURL + "/internal/health")
	if err != nil {
		return fmt.Errorf("health check: %w", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check: status=%d", resp.StatusCode)
	}
	return nil
}
