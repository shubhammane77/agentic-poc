package com.example.app;

public class GreetingService {
    public String greet(String name) {
        if (name == null || name.isBlank()) {
            return "Hello, stranger";
        }
        return "Hello, " + name;
    }
}
