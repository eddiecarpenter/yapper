//go:build tools

// Package tools pins library dependencies that are referenced from
// production code only in a later Feature task. Including the blank
// imports here — under the `tools` build constraint so the file is
// excluded from any normal build — keeps `go mod tidy` from
// stripping the corresponding require lines from go.mod between
// task commits.
//
// The pin for github.com/coder/websocket lives here because Task 1
// of Feature #15 bootstraps go.mod and Task 4 is the first task to
// actually import the package. Once Task 4 lands, this file may be
// removed; until then it locks the version transitive deps see.
package tools

import (
	_ "github.com/coder/websocket"
)
