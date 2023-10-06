#pragma once
#include "Dependencies/glm/glm.hpp"
#include "Dependencies/glew/glew.h"
using namespace glm;

//color
class Color {
private:
    void TidyRGBA() {
        r = r > 1.0f ? 1.0f : r;
        g = g > 1.0f ? 1.0f : g;
        b = b > 1.0f ? 1.0f : b;
        a = a > 1.0f ? 1.0f : a;
        r = r < 0.0f ? 0.0f : r;
        g = g < 0.0f ? 0.0f : g;
        b = b < 0.0f ? 0.0f : b;
        a = a < 0.0f ? 0.0f : a;
    }
public:
    float r = 0.0, g = 0.0, b = 0.0, a = 1.0;
    Color() {}
    Color(float r, float g, float b, float a = 1.0f) :r(r), g(g), b(b), a(a) {}
    Color(const vec4 color) :r(color.r), g(color.g), b(color.b), a(color.a) { TidyRGBA(); }
    Color(const vec3 color) :r(color.r), g(color.g), b(color.b), a(1.0f) { TidyRGBA(); }
    Color(const vec3 color, float a) :r(color.r), g(color.g), b(color.b), a(a) { TidyRGBA(); }
    Color(const vec4 color, float a) :r(color.r), g(color.g), b(color.b), a(a) { TidyRGBA(); }
    Color(int r, int g, int b, int a = 255) :r(r / 255.0f), g(g / 255.0f), b(b / 255.0f), a(a / 255.0f) { TidyRGBA(); }
    Color(const Color& color) :r(color.r), g(color.g), b(color.b), a(color.a) {}
    Color(const Color& color, float a) :r(color.r), g(color.g), b(color.b), a(a) { TidyRGBA(); }
    vec3 rgb = ([this]() { return vec3(r, g, b); })();
    vec4 rgba = ([this]() { return vec4(r, g, b, a); })();
    vec4 argb = ([this]() { return vec4(a, r, g, b); })();

    vec3 GetHSV() {
        float h, s, v;
        float max = glm::max(r, glm::max(g, b));
        float min = glm::min(r, glm::min(g, b));
        if (max == min) h = 0;
        else if (max == r) h = 60 * (g - b) / (max - min);
        else if (max == g) h = 60 * (b - r) / (max - min) + 120;
        else if (max == b) h = 60 * (r - g) / (max - min) + 240;
        if (h < 0) h += 360;
        s = max == 0 ? 0 : (max - min) / max;
        v = max;
        return vec3(h, s, v);
    }
    void SetColorFromHSV(vec3 hsv, float a = 1.0) {
        float h = hsv.x, s = hsv.y, v = hsv.z;
        float r, g, b;
        if (s == 0) {
            r = g = b = v;
        }
        else {
            int i = (int)h / 60;
            float f = h / 60 - i;
            float p = v * (1 - s);
            float q = v * (1 - s * f);
            float t = v * (1 - s * (1 - f));
            switch (i) {
            case 0:
                r = v;
                g = t;
                b = p;
                break;
            case 1:
                r = q;
                g = v;
                b = p;
                break;
            case 2:
                r = p;
                g = v;
                b = t;
                break;
            case 3:
                r = p;
                g = q;
                b = v;
                break;
            case 4:
                r = t;
                g = p;
                b = v;
                break;
            case 5:
                r = v;
                g = p;
                b = q;
                break;
            }
        }
    }
    Color& operator=(const Color& color) {
        r = color.r;
        g = color.g;
        b = color.b;
        a = color.a;
        return *this;
    }
    Color& operator=(const vec3& color) {
        r = color.r;
        g = color.g;
        b = color.b;
        TidyRGBA();
        return *this;
    }
    Color& operator=(const vec4& color) {
        r = color.r;
        g = color.g;
        b = color.b;
        a = color.a;
        TidyRGBA();
        return *this;
    }
    Color& operator=(const float& color) {
        r = color;
        g = color;
        b = color;
        TidyRGBA();
        return *this;
    }
    Color& operator=(const int& color) {
        r = color / 255.0f;
        g = color / 255.0f;
        b = color / 255.0f;
        TidyRGBA();
        return *this;
    }
    Color& operator+=(Color& color) {
        r += color.r;
        g += color.g;
        b += color.b;
        a += color.a;
        TidyRGBA();
        return *this;
    }
    Color& operator+=(vec3& color) {
        r += color.r;
        g += color.g;
        b += color.b;
        TidyRGBA();
        return *this;
    }
    Color& operator+=(vec4& color) {
        r += color.r;
        g += color.g;
        b += color.b;
        a += color.a;
        TidyRGBA();
        return *this;
    }
    Color& operator-=(Color& color) {
        r -= color.r;
        g -= color.g;
        b -= color.b;
        a -= color.a;
        TidyRGBA();
        return *this;
    }
    Color& operator-=(vec3& color) {
        r -= color.r;
        g -= color.g;
        b -= color.b;
        TidyRGBA();
        return *this;
    }
    Color& operator-=(vec4& color) {
        r -= color.r;
        g -= color.g;
        b -= color.b;
        a -= color.a;
        TidyRGBA();
        return *this;
    }
    Color& operator*=(Color& color) {
        r *= color.r;
        g *= color.g;
        b *= color.b;
        a *= color.a;
        TidyRGBA();
        return *this;
    }
    Color& operator*=(vec3& color) {
        r *= color.r;
        g *= color.g;
        b *= color.b;
        TidyRGBA();
        return *this;
    }
    Color& operator*=(vec4& color) {
        r *= color.r;
        g *= color.g;
        b *= color.b;
        a *= color.a;
        TidyRGBA();
        return *this;
    }
    Color& operator/=(Color& color) {
        r /= color.r;
        g /= color.g;
        b /= color.b;
        a /= color.a;
        TidyRGBA();
        return *this;
    }
    Color& operator/=(vec3& color) {
        r /= color.r;
        g /= color.g;
        b /= color.b;
        TidyRGBA();
        return *this;
    }
    Color& operator/=(vec4& color) {
        r /= color.r;
        g /= color.g;
        b /= color.b;
        a /= color.a;
        TidyRGBA();
        return *this;
    }
    Color& operator*=(float f) {
        r *= f;
        g *= f;
        b *= f;
        a *= f;
        TidyRGBA();
        return *this;
    }
    Color& operator/=(float f) {
        r /= f;
        g /= f;
        b /= f;
        a /= f;
        TidyRGBA();
        return *this;
    }
    Color operator+(Color& color) {
        return Color(r + color.r, g + color.g, b + color.b, a + color.a);
    }
    Color operator-(Color& color) {
        return Color(r - color.r, g - color.g, b - color.b, a - color.a);
    }
    Color operator*(Color& color) {
        return Color(r * color.r, g * color.g, b * color.b, a * color.a);
    }
    Color operator/(Color& color) {
        return Color(r / color.r, g / color.g, b / color.b, a / color.a);
    }
    Color operator*(float f) {
        return Color(r * f, g * f, b * f, a * f);
    }
    Color operator/(float f) {
        return Color(r / f, g / f, b / f, a / f);
    }
    bool operator==(Color& color) {
        return r == color.r && g == color.g && b == color.b && a == color.a;
    }
    bool operator!=(Color& color) {
        return r != color.r || g != color.g || b != color.b || a != color.a;
    }

    //constant colors
    static const Color Black;
    static const Color White;
    static const Color Red;
    static const Color Green;
    static const Color Blue;
    static const Color Yellow;
    static const Color Orange;
    static const Color Purple;
    static const Color Cyan;
    static const Color Magenta;
    static const Color Gray;
    static const Color Clear;
};
