package handler

import (
	"net/http"

	"github.com/gin-gonic/gin"

	"superassist-go/service"
)

type AuthHandler struct {
	svc *service.AuthService
}

func NewAuthHandler(svc *service.AuthService) *AuthHandler {
	return &AuthHandler{svc: svc}
}

type registerRequest struct {
	Username string `json:"username" binding:"required"`
	Password string `json:"password" binding:"required"`
}

type loginRequest struct {
	Username string `json:"username" binding:"required"`
	Password string `json:"password" binding:"required"`
}

// POST /api/auth/register
func (h *AuthHandler) Register(c *gin.Context) {
	var req registerRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusUnprocessableEntity, gin.H{"detail": "username and password are required"})
		return
	}

	token, user, err := h.svc.Register(req.Username, req.Password)
	if err != nil {
		switch err {
		case service.ErrUsernameTaken:
			c.JSON(http.StatusConflict, gin.H{"detail": err.Error()})
		case service.ErrUsernameTooShort, service.ErrPasswordTooShort:
			c.JSON(http.StatusUnprocessableEntity, gin.H{"detail": err.Error()})
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"detail": "registration failed"})
		}
		return
	}

	c.JSON(http.StatusCreated, gin.H{
		"user_id":      user.ID,
		"username":     user.Username,
		"is_admin":     user.IsAdmin,
		"access_token": token,
		"token_type":   "bearer",
	})
}

// POST /api/auth/login
func (h *AuthHandler) Login(c *gin.Context) {
	var req loginRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusUnprocessableEntity, gin.H{"detail": "username and password are required"})
		return
	}

	token, user, err := h.svc.Login(req.Username, req.Password)
	if err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"detail": "invalid username or password"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"user_id":      user.ID,
		"username":     user.Username,
		"is_admin":     user.IsAdmin,
		"access_token": token,
		"token_type":   "bearer",
	})
}

// GET /api/auth/me
func (h *AuthHandler) Me(c *gin.Context) {
	userID := c.GetString("user_id")
	user, err := h.svc.GetUser(userID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"detail": "user not found"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"user_id":  user.ID,
		"username": user.Username,
		"is_admin": user.IsAdmin,
	})
}
