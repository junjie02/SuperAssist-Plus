package handler

import (
	"encoding/json"
	"net/http"

	"github.com/gin-gonic/gin"

	"superassist-go/proxy"
)

type GraphHandler struct {
	client *proxy.PythonClient
}

func NewGraphHandler(client *proxy.PythonClient) *GraphHandler {
	return &GraphHandler{client: client}
}

// GET /api/graph — proxy to Python AI engine.
func (h *GraphHandler) Get(c *gin.Context) {
	userID := c.GetString("user_id")

	raw, err := h.client.GraphPayload(userID)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"detail": "failed to fetch graph from AI engine"})
		return
	}

	// Decode and re-encode to validate JSON
	var payload json.RawMessage
	if err := json.Unmarshal(raw, &payload); err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"detail": "invalid graph response from AI engine"})
		return
	}

	c.Data(http.StatusOK, "application/json", raw)
}
