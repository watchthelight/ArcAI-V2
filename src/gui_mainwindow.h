#ifndef GUI_MAINWINDOW_H
#define GUI_MAINWINDOW_H

#include <QMainWindow>
#include <QPushButton>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QProcess>
#include <QFileDialog>
#include <QDialog>
#include <QFormLayout>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QLineEdit>
#include <QDialogButtonBox>

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void selectDataFile();
    void startTraining();
    void startGeneration();
    void readProcessOutput();
    void processFinished(int exitCode, QProcess::ExitStatus exitStatus);

private:
    void setupUI();
    void setupProcess();
    QString selectedDataFile;
    QString datasetPath;

    // UI elements
    QWidget *centralWidget;
    QHBoxLayout *mainLayout;

    // Left panel
    QGroupBox *controlGroup;
    QVBoxLayout *controlLayout;
    QPushButton *selectDataButton;
    QPushButton *trainButton;
    QPushButton *generateButton;

    // Right panel
    QGroupBox *outputGroup;
    QVBoxLayout *outputLayout;
    QTextEdit *outputText;

    // Process
    QProcess *process;
};

class GenerationDialog : public QDialog {
    Q_OBJECT

public:
    GenerationDialog(QWidget *parent = nullptr);

    int getLength() const { return lengthSpin->value(); }
    double getTemperature() const { return tempSpin->value(); }
    int getTopK() const { return topKSpin->value(); }
    QString getSeed() const { return seedEdit->text(); }

private:
    QSpinBox *lengthSpin;
    QDoubleSpinBox *tempSpin;
    QSpinBox *topKSpin;
    QLineEdit *seedEdit;
};

#endif // GUI_MAINWINDOW_H
