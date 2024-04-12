#include <fmt/color.h>
#include "spdlog/spdlog.h"
#include <nlohmann/json.hpp>


int main() {
    fmt::print(fg(fmt::color::crimson) | fmt::emphasis::bold, "Hello, {}!\n", "fmt");

    spdlog::info("Hello, spdlog");

    nlohmann::json a = {
        { "key", "value" }
    };
    
    return 0;
}