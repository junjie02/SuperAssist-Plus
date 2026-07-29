package config

import (
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/joho/godotenv"
)

// projectRoot is the directory containing .env (the repo root).
// All relative paths are resolved from here so that Go and Python
// share the same data directory.
var projectRoot string

// Config holds all runtime settings parsed from environment variables.
type Config struct {
	// Database
	DBURL  string // empty = SQLite fallback
	DBPath string // SQLite path (when DBURL is empty)

	// Server
	GoPort     string
	PythonHost string // Python AI engine address

	// JWT
	JWTSecret      string
	JWTExpiryHours int
	AdminUsernames map[string]struct{}

	// Data
	DataDir string
}

func Load() *Config {
	// --- Resolve project root from .env location -----------------------
	projectRoot, _ = os.Getwd()
	for _, candidate := range []string{
		".env",       // cwd is project root
		"../.env",    // cwd is go-server/
		"../../.env", // cwd is deeper
	} {
		if _, err := os.Stat(candidate); err == nil {
			if err := godotenv.Load(candidate); err == nil {
				log.Printf("config: loaded %s", candidate)
				// Set projectRoot to the directory containing .env
				if abs, err := filepath.Abs(filepath.Dir(candidate)); err == nil {
					projectRoot = abs
				}
			}
			break
		}
	}
	log.Printf("config: project root = %s", projectRoot)

	// --- Read env vars ------------------------------------------------
	cfg := &Config{
		DBURL:          getEnv("SUPERASSIST_DB_URL", ""),
		GoPort:         getEnv("SUPERASSIST_GO_PORT", "8080"),
		PythonHost:     getEnv("SUPERASSIST_PYTHON_HOST", "http://127.0.0.1:8765"),
		JWTSecret:      getEnv("SUPERASSIST_JWT_SECRET", ""),
		JWTExpiryHours: getEnvInt("SUPERASSIST_JWT_EXPIRY_HOURS", 48),
		AdminUsernames: parseSet(getEnv("SUPERASSIST_ADMIN_USERNAMES", "painting")),
		DataDir:        getEnv("SUPERASSIST_DATA_DIR", ".superassist"),
	}

	// --- Resolve relative paths from project root ---------------------
	if !filepath.IsAbs(cfg.DataDir) {
		cfg.DataDir = filepath.Join(projectRoot, cfg.DataDir)
	}
	cfg.DBPath = filepath.Join(cfg.DataDir, "superassist.sqlite3")

	if cfg.JWTSecret == "" {
		cfg.JWTSecret = "superassist-dev-secret-change-in-production"
	}

	return cfg
}

func (c *Config) IsAdminUsername(username string) bool {
	_, ok := c.AdminUsernames[strings.ToLower(strings.TrimSpace(username))]
	return ok
}

func parseSet(value string) map[string]struct{} {
	result := make(map[string]struct{})
	for _, item := range strings.Split(value, ",") {
		item = strings.ToLower(strings.TrimSpace(item))
		if item != "" {
			result[item] = struct{}{}
		}
	}
	return result
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func getEnvInt(key string, fallback int) int {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	var n int
	for _, c := range v {
		if c < '0' || c > '9' {
			return fallback
		}
		n = n*10 + int(c-'0')
	}
	return n
}
