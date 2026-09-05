package main

import (
	"fmt"
)

func main() {
	intArr := [3]int32{1,4,5}
	intArrB := [...]int32{1,4,5}
	fmt.Println(intArr)
	fmt.Println(intArrB)
}
