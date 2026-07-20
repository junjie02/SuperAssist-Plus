package proxy

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
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
}

// Chat sends a message to the Python AI engine and returns the SSE response body.
// The caller is responsible for closing the body.
func (p *PythonClient) Chat(userID, threadID, message string) (*http.Response, error) {
	reqBody := ChatRequest{
		UserID:   userID,
		Message:  message,
		ThreadID: threadID,
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
		}
		ch <- evt
	}
	return scanner.Err()
}

// GraphPayload fetches the memory graph from Python.
func (p *PythonClient) GraphPayload(userID string) (json.RawMessage, error) {
	url := fmt.Sprintf("%s/internal/graph?user_id=%s", p.baseURL, userID)
	resp, err := p.client.Get(url)
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
