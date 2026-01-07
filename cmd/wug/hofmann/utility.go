package main

import (
	"slices"

	"golang.org/x/exp/constraints"
)

func getKeys[K constraints.Ordered, V any](m map[K]V) []K {
	keys := make([]K, 0, len(m))

	for k := range m {
		keys = append(keys, k)
	}

	slices.Sort(keys)

	return keys
}

func toSet[T comparable](slice []T) map[T]struct{} {
	s := make(map[T]struct{}, len(slice))

	for _, v := range slice {
		s[v] = struct{}{}
	}

	return s
}
