package model

import (
	"crypto/rand"
	"encoding/hex"
	"time"
)

// User represents a registered user.
type User struct {
	ID           string    `gorm:"primaryKey;size:64" json:"id"`
	Username     string    `gorm:"uniqueIndex;size:128;not null" json:"username"`
	PasswordHash string    `gorm:"size:256;not null" json:"-"`
	CreatedAt    time.Time `json:"created_at"`
	UpdatedAt    time.Time `json:"updated_at"`
}

func NewUserID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return "user_" + hex.EncodeToString(b)[:12]
}
