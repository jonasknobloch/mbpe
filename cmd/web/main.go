//go:build wasm

package main

import (
	"syscall/js"

	"github.com/jonasknobloch/mbpe/internal/web"
)

func main() {
	js.Global().Set("tokenizeWeb", web.WrapTokenizeWeb())

	// Keep the Go runtime alive
	select {}
}
