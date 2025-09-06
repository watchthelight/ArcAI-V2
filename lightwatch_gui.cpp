#include <QApplication>
#include <QDir>
#include "gui_mainwindow.h"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    // Set application properties
    app.setApplicationName("LightwatchAI");
    app.setApplicationVersion("2.0");
    app.setOrganizationName("LightwatchAI");

    // Set working directory to executable directory
    QDir::setCurrent(QApplication::applicationDirPath());

    MainWindow window;
    window.show();

    return app.exec();
}
