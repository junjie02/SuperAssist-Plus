package service

import (
	"errors"
	"fmt"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"golang.org/x/crypto/bcrypt"
	"gorm.io/gorm"

	"superassist-go/config"
	"superassist-go/model"
)

type AuthService struct {
	db  *gorm.DB
	cfg *config.Config
}

func NewAuthService(db *gorm.DB, cfg *config.Config) *AuthService {
	return &AuthService{db: db, cfg: cfg}
}

var (
	ErrUsernameTaken    = errors.New("username already taken")
	ErrInvalidCreds     = errors.New("invalid username or password")
	ErrUsernameTooShort = errors.New("username must be at least 2 characters")
	ErrPasswordTooShort = errors.New("password must be at least 6 characters")
)

// Register creates a new user and returns a JWT token.
func (s *AuthService) Register(username, password string) (string, *model.User, error) {
	if len(username) < 2 {
		return "", nil, ErrUsernameTooShort
	}
	if len(password) < 6 {
		return "", nil, ErrPasswordTooShort
	}

	// Check for existing user
	var existing model.User
	if err := s.db.Where("username = ?", username).First(&existing).Error; err == nil {
		return "", nil, ErrUsernameTaken
	}

	hash, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return "", nil, fmt.Errorf("hash password: %w", err)
	}

	user := model.User{
		ID:           model.NewUserID(),
		Username:     username,
		PasswordHash: string(hash),
		CreatedAt:    time.Now().UTC(),
		UpdatedAt:    time.Now().UTC(),
	}

	if err := s.db.Create(&user).Error; err != nil {
		return "", nil, fmt.Errorf("create user: %w", err)
	}

	token, err := s.createToken(user.ID)
	if err != nil {
		return "", nil, err
	}

	return token, &user, nil
}

// Login verifies credentials and returns a JWT token.
func (s *AuthService) Login(username, password string) (string, *model.User, error) {
	var user model.User
	if err := s.db.Where("username = ?", username).First(&user).Error; err != nil {
		return "", nil, ErrInvalidCreds
	}

	if err := bcrypt.CompareHashAndPassword([]byte(user.PasswordHash), []byte(password)); err != nil {
		return "", nil, ErrInvalidCreds
	}

	token, err := s.createToken(user.ID)
	if err != nil {
		return "", nil, err
	}

	return token, &user, nil
}

// GetUser returns a user by ID.
func (s *AuthService) GetUser(userID string) (*model.User, error) {
	var user model.User
	if err := s.db.Where("id = ?", userID).First(&user).Error; err != nil {
		return nil, err
	}
	return &user, nil
}

func (s *AuthService) createToken(userID string) (string, error) {
	claims := jwt.MapClaims{
		"sub": userID,
		"exp": time.Now().Add(time.Duration(s.cfg.JWTExpiryHours) * time.Hour).Unix(),
		"iat": time.Now().Unix(),
	}
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString([]byte(s.cfg.JWTSecret))
}
