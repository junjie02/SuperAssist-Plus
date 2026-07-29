package middleware

import (
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt/v5"
	"gorm.io/gorm"

	"superassist-go/model"
)

// JWTAuth returns a Gin middleware that validates Bearer tokens.
func JWTAuth(secret string) gin.HandlerFunc {
	return func(c *gin.Context) {
		auth := c.GetHeader("Authorization")
		if auth == "" {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"detail": "Missing Authorization header"})
			return
		}

		parts := strings.SplitN(auth, " ", 2)
		if len(parts) != 2 || !strings.EqualFold(parts[0], "Bearer") {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"detail": "Invalid Authorization header format"})
			return
		}

		token, err := jwt.Parse(parts[1], func(t *jwt.Token) (any, error) {
			if _, ok := t.Method.(*jwt.SigningMethodHMAC); !ok {
				return nil, jwt.ErrSignatureInvalid
			}
			return []byte(secret), nil
		})
		if err != nil || !token.Valid {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"detail": "Invalid or expired token"})
			return
		}

		claims, ok := token.Claims.(jwt.MapClaims)
		if !ok {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"detail": "Invalid token claims"})
			return
		}

		userID, _ := claims["sub"].(string)
		if userID == "" {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"detail": "Invalid token subject"})
			return
		}

		c.Set("user_id", userID)
		c.Next()
	}
}

// RequireAdmin rejects authenticated users without the persisted admin role.
func RequireAdmin(db *gorm.DB) gin.HandlerFunc {
	return func(c *gin.Context) {
		var user model.User
		if err := db.Select("id", "is_admin").Where("id = ?", c.GetString("user_id")).First(&user).Error; err != nil {
			c.AbortWithStatusJSON(http.StatusForbidden, gin.H{"detail": "administrator access required"})
			return
		}
		if !user.IsAdmin {
			c.AbortWithStatusJSON(http.StatusForbidden, gin.H{"detail": "administrator access required"})
			return
		}
		c.Next()
	}
}

// ExtractUserIDFromQuery extracts JWT from ?token= query param (for WebSocket).
func ExtractUserIDFromQuery(secret string, tokenStr string) (string, error) {
	token, err := jwt.Parse(tokenStr, func(t *jwt.Token) (any, error) {
		if _, ok := t.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, jwt.ErrSignatureInvalid
		}
		return []byte(secret), nil
	})
	if err != nil || !token.Valid {
		return "", err
	}

	claims, ok := token.Claims.(jwt.MapClaims)
	if !ok {
		return "", jwt.ErrSignatureInvalid
	}

	userID, _ := claims["sub"].(string)
	if userID == "" {
		return "", jwt.ErrSignatureInvalid
	}

	return userID, nil
}
