package handler

import (
	"net/http"
	"os"
	"path/filepath"

	"github.com/gin-gonic/gin"
)

// ServeSPA serves the React SPA with fallback to index.html.
// frontendDir is the path to the React build output.
func ServeSPA(frontendDir string) gin.HandlerFunc {
	return func(c *gin.Context) {
		// If the request has a file extension, try to serve it directly
		ext := filepath.Ext(c.Request.URL.Path)
		if ext != "" {
			target := filepath.Join(frontendDir, c.Request.URL.Path)
			if _, err := os.Stat(target); err == nil {
				c.File(target)
				return
			}
			c.Status(http.StatusNotFound)
			return
		}

		// SPA fallback: serve index.html for all non-file routes
		indexPath := filepath.Join(frontendDir, "index.html")
		if _, err := os.Stat(indexPath); err != nil {
			c.String(http.StatusOK, "SuperAssist — frontend not built. Run: cd frontend && npm run build")
			return
		}
		c.File(indexPath)
	}
}
