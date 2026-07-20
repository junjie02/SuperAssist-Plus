package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/glebarez/sqlite"
	"github.com/gin-gonic/gin"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"

	"superassist-go/config"
	"superassist-go/handler"
	"superassist-go/middleware"
	"superassist-go/model"
	"superassist-go/proxy"
	"superassist-go/service"
	"superassist-go/ws"
)

func main() {
	cfg := config.Load()

	// ---- Database ----------------------------------------------------------
	db, err := openDB(cfg)
	if err != nil {
		log.Fatalf("Failed to open database: %v", err)
	}
	if err := db.AutoMigrate(&model.User{}); err != nil {
		log.Fatalf("Failed to migrate database: %v", err)
	}

	// ---- Services ----------------------------------------------------------
	authSvc := service.NewAuthService(db, cfg)
	pythonClient := proxy.NewPythonClient(cfg.PythonHost)

	// ---- Handlers ----------------------------------------------------------
	authH := handler.NewAuthHandler(authSvc)
	graphH := handler.NewGraphHandler(pythonClient)
	threadH := handler.NewThreadHandler(db, cfg.DataDir)
	chatH := ws.NewChatHandler(pythonClient, cfg.JWTSecret)

	// ---- Router ------------------------------------------------------------
	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()

	// Public routes (no auth required)
	auth := r.Group("/api/auth")
	{
		auth.POST("/register", authH.Register)
		auth.POST("/login", authH.Login)
	}

	// Protected routes (auth required)
	api := r.Group("/api")
	api.Use(middleware.JWTAuth(cfg.JWTSecret))
	{
		api.GET("/auth/me", authH.Me)
		api.GET("/threads", threadH.GetThreads)
		api.GET("/threads/:id/history", threadH.GetHistory)
		api.DELETE("/threads/:id", threadH.DeleteThread)
		api.GET("/graph", graphH.Get)
	}

	// WebSocket (auth via query param)
	r.GET("/ws/chat", chatH.Handle)

	// Health (public)
	r.GET("/api/internal/health", func(c *gin.Context) {
		if err := pythonClient.Health(); err != nil {
			c.JSON(503, gin.H{"status": "degraded", "python": err.Error()})
			return
		}
		c.JSON(200, gin.H{"status": "ok"})
	})

	// SPA — serve React frontend
	frontendDir := findFrontendDir()
	r.NoRoute(handler.ServeSPA(frontendDir))

	// ---- Start -------------------------------------------------------------
	addr := fmt.Sprintf(":%s", cfg.GoPort)
	log.Printf("SuperAssist Go server starting on http://localhost%s", addr)
	log.Printf("  Python AI engine: %s", cfg.PythonHost)
	log.Printf("  Frontend: %s", frontendDir)
	if cfg.DBURL == "" {
		log.Printf("  Database: SQLite at %s", cfg.DBPath)
	} else {
		log.Printf("  Database: MySQL (%s)", cfg.DBURL)
	}

	if err := r.Run(addr); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}

func openDB(cfg *config.Config) (*gorm.DB, error) {
	if cfg.DBURL != "" {
		return gorm.Open(mysql.Open(cfg.DBURL), &gorm.Config{})
	}

	// SQLite fallback
	dir := filepath.Dir(cfg.DBPath)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("create data dir: %w", err)
	}
	return gorm.Open(sqlite.Open(cfg.DBPath), &gorm.Config{})
}

func findFrontendDir() string {
	// Try common locations
	candidates := []string{
		"../frontend/dist", // from go-server/ to frontend/dist
		"frontend/dist",    // from project root
	}
	for _, d := range candidates {
		if info, err := os.Stat(d); err == nil && info.IsDir() {
			abs, _ := filepath.Abs(d)
			return abs
		}
	}
	// Fallback: relative from go-server directory
	return "../frontend/dist"
}
