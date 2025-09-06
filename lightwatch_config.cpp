#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#include <conio.h>
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#include <termios.h>
#include <sys/ioctl.h>
#endif

struct LightwatchConfig {
    int hidden_size;
    int tbptt_len;
};

struct MenuOption {
    std::string display_name;
    int value;
};

// Model size options
static const std::vector<MenuOption> MODEL_SIZE_OPTIONS = {
    {"Minimum", 64},
    {"Mini", 96},
    {"NotSoMini", 128},
    {"Normaal", 160},
    {"Normalish", 192},
    {"Big", 224},
    {"Bigger", 256},
    {"Biggest", 320},
    {"LiterallyInsane", 512}
};

// TBPTT length options
static const std::vector<MenuOption> TBPTT_OPTIONS = {
    {"Shortest", 1},
    {"Short", 8},
    {"NotShortButNotLong", 16},
    {"YesLong", 32},
    {"Longer", 64},
    {"Longest", 128}
};

// Cross-platform terminal utilities
class TerminalUtils {
public:
    static bool isInteractive() {
#ifdef _WIN32
        return _isatty(_fileno(stdin)) && _isatty(_fileno(stdout));
#else
        return isatty(STDIN_FILENO) && isatty(STDOUT_FILENO);
#endif
    }

    static void clearScreen() {
#ifdef _WIN32
        system("cls");
#else
        system("clear");
#endif
    }

    static int getKey() {
#ifdef _WIN32
        return _getch();
#else
        struct termios oldt, newt;
        tcgetattr(STDIN_FILENO, &oldt);
        newt = oldt;
        newt.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);
        int ch = getchar();
        tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
        return ch;
#endif
    }

    static void hideCursor() {
#ifdef _WIN32
        CONSOLE_CURSOR_INFO cursorInfo;
        GetConsoleCursorInfo(GetStdHandle(STD_OUTPUT_HANDLE), &cursorInfo);
        cursorInfo.bVisible = false;
        SetConsoleCursorInfo(GetStdHandle(STD_OUTPUT_HANDLE), &cursorInfo);
#else
        std::cout << "\033[?25l" << std::flush;
#endif
    }

    static void showCursor() {
#ifdef _WIN32
        CONSOLE_CURSOR_INFO cursorInfo;
        GetConsoleCursorInfo(GetStdHandle(STD_OUTPUT_HANDLE), &cursorInfo);
        cursorInfo.bVisible = true;
        SetConsoleCursorInfo(GetStdHandle(STD_OUTPUT_HANDLE), &cursorInfo);
#else
        std::cout << "\033[?25h" << std::flush;
#endif
    }
};

// Interactive menu selection
static int selectFromMenu(const std::string& title, const std::vector<MenuOption>& options, int defaultIndex = 0) {
    if (!TerminalUtils::isInteractive()) {
        return options[defaultIndex].value;
    }

    int selected = defaultIndex;
    TerminalUtils::hideCursor();

    while (true) {
        TerminalUtils::clearScreen();
        
        std::cout << "=== LightwatchAI Configuration ===" << std::endl;
        std::cout << title << std::endl;
        std::cout << std::endl;
        std::cout << "Use ↑/↓ arrow keys to navigate, Enter to select:" << std::endl;
        std::cout << std::endl;

        for (size_t i = 0; i < options.size(); ++i) {
            if (i == static_cast<size_t>(selected)) {
                std::cout << "→ " << options[i].display_name << " (" << options[i].value << ")" << std::endl;
            } else {
                std::cout << "  " << options[i].display_name << " (" << options[i].value << ")" << std::endl;
            }
        }

        std::cout << std::endl;
        std::cout << "Press 'q' to quit" << std::endl;

        int key = TerminalUtils::getKey();

#ifdef _WIN32
        if (key == 224) { // Arrow key prefix on Windows
            key = TerminalUtils::getKey();
            if (key == 72) { // Up arrow
                selected = (selected - 1 + options.size()) % options.size();
            } else if (key == 80) { // Down arrow
                selected = (selected + 1) % options.size();
            }
        }
#else
        if (key == 27) { // ESC sequence
            key = TerminalUtils::getKey();
            if (key == 91) { // [
                key = TerminalUtils::getKey();
                if (key == 65) { // Up arrow
                    selected = (selected - 1 + options.size()) % options.size();
                } else if (key == 66) { // Down arrow
                    selected = (selected + 1) % options.size();
                }
            }
        }
#endif
        else if (key == 13 || key == 10) { // Enter
            break;
        } else if (key == 'q' || key == 'Q') {
            TerminalUtils::showCursor();
            std::exit(0);
        }
    }

    TerminalUtils::showCursor();
    return options[selected].value;
}

// Parse command line arguments
static LightwatchConfig parseCommandLine(int argc, char** argv) {
    LightwatchConfig config = {128, 32}; // defaults

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--hidden-size") == 0 && i + 1 < argc) {
            config.hidden_size = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--tbptt") == 0 && i + 1 < argc) {
            config.tbptt_len = std::atoi(argv[++i]);
        }
    }

    return config;
}

// Main configuration function
LightwatchConfig getLightwatchConfig(int argc, char** argv) {
    // Check for command line overrides first
    LightwatchConfig config = parseCommandLine(argc, argv);
    
    // If we have command line overrides, use them
    bool hasHiddenOverride = false;
    bool hasTbpttOverride = false;
    
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--hidden-size") == 0) hasHiddenOverride = true;
        if (std::strcmp(argv[i], "--tbptt") == 0) hasTbpttOverride = true;
    }

    // Interactive configuration if no overrides and terminal is interactive
    if (TerminalUtils::isInteractive()) {
        if (!hasHiddenOverride) {
            // Find default index for current hidden size
            int defaultIdx = 2; // NotSoMini (128) as default
            for (size_t i = 0; i < MODEL_SIZE_OPTIONS.size(); ++i) {
                if (MODEL_SIZE_OPTIONS[i].value == config.hidden_size) {
                    defaultIdx = i;
                    break;
                }
            }
            config.hidden_size = selectFromMenu("Select Model Size (Hidden Layer Size):", MODEL_SIZE_OPTIONS, defaultIdx);
        }

        if (!hasTbpttOverride) {
            // Find default index for current TBPTT length
            int defaultIdx = 3; // YesLong (32) as default
            for (size_t i = 0; i < TBPTT_OPTIONS.size(); ++i) {
                if (TBPTT_OPTIONS[i].value == config.tbptt_len) {
                    defaultIdx = i;
                    break;
                }
            }
            config.tbptt_len = selectFromMenu("Select TBPTT Length (Sequence Length):", TBPTT_OPTIONS, defaultIdx);
        }

        // Show final configuration
        TerminalUtils::clearScreen();
        std::cout << "=== LightwatchAI Configuration Complete ===" << std::endl;
        std::cout << "Hidden Size: " << config.hidden_size << std::endl;
        std::cout << "TBPTT Length: " << config.tbptt_len << std::endl;
        std::cout << std::endl;
        std::cout << "Press any key to continue..." << std::endl;
        TerminalUtils::getKey();
        TerminalUtils::clearScreen();
    }

    return config;
}

// Validation function
bool validateLightwatchConfig(const LightwatchConfig& config) {
    if (config.hidden_size < 32 || config.hidden_size > 1024) {
        std::cerr << "Error: Hidden size must be between 32 and 1024" << std::endl;
        return false;
    }
    
    if (config.tbptt_len < 1 || config.tbptt_len > 256) {
        std::cerr << "Error: TBPTT length must be between 1 and 256" << std::endl;
        return false;
    }
    
    return true;
}
