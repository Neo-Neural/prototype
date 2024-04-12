#include <fmt/color.h>
#include "spdlog/spdlog.h"

int main() {
    fmt::print(fg(fmt::color::crimson) | fmt::emphasis::bold, "Hello, {}!\n", "fmt");

    spdlog::info("Hello, spdlog");
    return 0;
}