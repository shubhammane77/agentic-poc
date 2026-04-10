package com.example;

public class Calculator {
    public int add(int left, int right) {
        return left + right;
    }

    public int subtract(int left, int right) {
        return left - right;
    }

    public int absoluteDifference(int left, int right) {
        if (left > right) {
            return left - right;
        }
        return right - left;
    }
}
