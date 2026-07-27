package handler

import (
	"encoding/json"
	"io"
	"net/http"

	"github.com/gin-gonic/gin"

	"superassist-go/proxy"
)

type SettingsHandler struct {
	client *proxy.PythonClient
}

func NewSettingsHandler(client *proxy.PythonClient) *SettingsHandler {
	return &SettingsHandler{client: client}
}

func (h *SettingsHandler) Get(c *gin.Context) {
	raw, err := h.client.SettingsPayload()
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"detail": "failed to fetch settings from AI engine"})
		return
	}
	c.Data(http.StatusOK, "application/json", raw)
}

func (h *SettingsHandler) Update(c *gin.Context) {
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, 64<<10)
	raw, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"detail": "invalid settings payload"})
		return
	}
	if !json.Valid(raw) {
		c.JSON(http.StatusBadRequest, gin.H{"detail": "invalid JSON"})
		return
	}

	response, status, err := h.client.UpdateSettings(raw)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"detail": "failed to update settings in AI engine"})
		return
	}
	c.Data(status, "application/json", response)
}
