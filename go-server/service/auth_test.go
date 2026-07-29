package service

import (
	"testing"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"

	"superassist-go/config"
	"superassist-go/model"
)

func TestSyncConfiguredAdminsPromotesPainting(t *testing.T) {
	db, err := gorm.Open(sqlite.Open(":memory:"), &gorm.Config{})
	if err != nil {
		t.Fatal(err)
	}
	if err := db.AutoMigrate(&model.User{}); err != nil {
		t.Fatal(err)
	}
	user := model.User{ID: "user_1", Username: "painting", PasswordHash: "x"}
	if err := db.Create(&user).Error; err != nil {
		t.Fatal(err)
	}
	service := NewAuthService(db, &config.Config{AdminUsernames: map[string]struct{}{"painting": {}}})
	if err := service.SyncConfiguredAdmins(); err != nil {
		t.Fatal(err)
	}
	if err := db.First(&user, "id = ?", user.ID).Error; err != nil {
		t.Fatal(err)
	}
	if !user.IsAdmin {
		t.Fatal("painting was not promoted to administrator")
	}
}
