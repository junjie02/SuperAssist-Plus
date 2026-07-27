package ws

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"

	"superassist-go/middleware"
	"superassist-go/proxy"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

// ChatMessage is the JSON payload from the browser.
type ChatMessage struct {
	Message  string `json:"message"`
	ThreadID string `json:"thread_id,omitempty"`
	RAGMode  bool   `json:"rag_mode"`
}

// ChatHandler manages WebSocket chat connections.
type ChatHandler struct {
	client    *proxy.PythonClient
	jwtSecret string
}

func NewChatHandler(client *proxy.PythonClient, jwtSecret string) *ChatHandler {
	return &ChatHandler{client: client, jwtSecret: jwtSecret}
}

// Handle upgrades the HTTP connection to WebSocket and proxies chat messages
// to the Python AI engine via SSE, pushing events back to the browser.
func (h *ChatHandler) Handle(c *gin.Context) {
	// Auth via ?token= query param (browser WebSocket API does not support custom headers)
	tokenStr := c.Query("token")
	if tokenStr == "" {
		c.JSON(http.StatusUnauthorized, gin.H{"detail": "Missing token query parameter"})
		return
	}

	userID, err := middleware.ExtractUserIDFromQuery(h.jwtSecret, tokenStr)
	if err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"detail": "Invalid or expired token"})
		return
	}

	conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		log.Printf("ws: upgrade error: %v", err)
		return
	}
	defer conn.Close()

	log.Printf("ws: connected user=%s", userID)

	for {
		_, raw, err := conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseNormalClosure) {
				log.Printf("ws: read error: %v", err)
			}
			break
		}

		var msg ChatMessage
		if err := json.Unmarshal(raw, &msg); err != nil {
			conn.WriteJSON(gin.H{"type": "error", "message": "invalid JSON"})
			continue
		}

		if msg.Message == "" {
			continue
		}

		// Call Python AI engine
		resp, err := h.client.Chat(userID, msg.ThreadID, msg.Message, msg.RAGMode)
		if err != nil {
			log.Printf("ws: python error: %v", err)
			conn.WriteJSON(map[string]any{
				"type":    "error",
				"message": fmt.Sprintf("AI engine error: %v", err),
			})
			continue
		}

		// Read SSE events and push to WebSocket
		ch := make(chan proxy.SSEEvent)
		go proxy.ReadSSEEvents(resp.Body, ch)

		for evt := range ch {
			writeJSON(conn, evt)
		}
	}
}

func writeJSON(conn *websocket.Conn, evt proxy.SSEEvent) {
	var payload any
	if evt.Raw != nil {
		// Raw event — pass through as-is
		var raw map[string]any
		if err := json.Unmarshal(evt.Raw, &raw); err == nil {
			payload = raw
		}
	}
	if payload == nil {
		payload = map[string]any{
			"type":    evt.Type,
			"content": evt.Content,
		}
	}
	conn.WriteJSON(payload)
}
