package handler

import (
	"net/http"

	"github.com/gin-gonic/gin"

	"superassist-go/proxy"
)

const maxRAGUploadBytes int64 = 512 << 20

type RAGHandler struct {
	client *proxy.PythonClient
}

func NewRAGHandler(client *proxy.PythonClient) *RAGHandler {
	return &RAGHandler{client: client}
}

func (h *RAGHandler) List(c *gin.Context) {
	raw, status, err := h.client.RAGDocuments(c.GetString("user_id"))
	writeRAGResponse(c, raw, status, err)
}

func (h *RAGHandler) Graph(c *gin.Context) {
	raw, status, err := h.client.RAGGraph(c.GetString("user_id"))
	writeRAGResponse(c, raw, status, err)
}

func (h *RAGHandler) Upload(c *gin.Context) {
	contentType := c.GetHeader("Content-Type")
	if contentType == "" {
		c.JSON(http.StatusBadRequest, gin.H{"detail": "missing multipart content type"})
		return
	}
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxRAGUploadBytes)
	raw, status, err := h.client.UploadRAGDocuments(c.GetString("user_id"), contentType, c.Request.Body)
	writeRAGResponse(c, raw, status, err)
}

func (h *RAGHandler) Delete(c *gin.Context) {
	raw, status, err := h.client.DeleteRAGDocument(c.GetString("user_id"), c.Param("id"))
	writeRAGResponse(c, raw, status, err)
}

func writeRAGResponse(c *gin.Context, raw []byte, status int, err error) {
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"detail": "knowledge service is unavailable"})
		return
	}
	c.Data(status, "application/json", raw)
}
