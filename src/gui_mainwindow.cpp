#include "gui_mainwindow.h"
#include <QApplication>
#include <QMessageBox>
#include <QStandardPaths>
#include <QDir>
#include <QFileInfo>
#include <QTextStream>
#include <QDebug>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    setupUI();
    setupProcess();
    setWindowTitle("LightwatchAI GUI");
    resize(1000, 600);
}

MainWindow::~MainWindow() {
    if (process && process->state() != QProcess::NotRunning) {
        process->kill();
        process->waitForFinished(3000);
    }
}

void MainWindow::setupUI() {
    centralWidget = new QWidget;
    setCentralWidget(centralWidget);

    mainLayout = new QHBoxLayout(centralWidget);

    // Left panel - Controls
    controlGroup = new QGroupBox("Controls");
    controlLayout = new QVBoxLayout(controlGroup);

    selectDataButton = new QPushButton("Select Training Data");
    trainButton = new QPushButton("Train");
    generateButton = new QPushButton("Generate");

    controlLayout->addWidget(selectDataButton);
    controlLayout->addWidget(trainButton);
    controlLayout->addWidget(generateButton);
    controlLayout->addStretch();

    // Right panel - Output
    outputGroup = new QGroupBox("Output");
    outputLayout = new QVBoxLayout(outputGroup);

    outputText = new QTextEdit;
    outputText->setReadOnly(true);
    outputText->setFont(QFont("Courier New", 10));
    outputLayout->addWidget(outputText);

    // Add panels to main layout
    mainLayout->addWidget(controlGroup, 1);
    mainLayout->addWidget(outputGroup, 3);

    // Connect signals
    connect(selectDataButton, &QPushButton::clicked, this, &MainWindow::selectDataFile);
    connect(trainButton, &QPushButton::clicked, this, &MainWindow::startTraining);
    connect(generateButton, &QPushButton::clicked, this, &MainWindow::startGeneration);
}

void MainWindow::setupProcess() {
    process = new QProcess(this);
    connect(process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &MainWindow::processFinished);
    connect(process, &QProcess::readyReadStandardOutput, this, &MainWindow::readProcessOutput);
    connect(process, &QProcess::readyReadStandardError, this, &MainWindow::readProcessOutput);
}

void MainWindow::selectDataFile() {
    QString fileName = QFileDialog::getOpenFileName(this,
        "Select Training Data File",
        QStandardPaths::writableLocation(QStandardPaths::HomeLocation),
        "Text files (*.txt);;All files (*)");

    if (!fileName.isEmpty()) {
        selectedDataFile = fileName;
        datasetPath = QFileInfo(fileName).absolutePath() + "/dataset.bin";

        outputText->append("Selected data file: " + fileName);
        outputText->append("Dataset will be saved as: " + datasetPath);

        // Preprocess the data
        outputText->append("Preprocessing data...");

        QStringList args;
        args << fileName << datasetPath << "vocab.json";

        process->start("./data_preprocess", args);
    }
}

void MainWindow::startTraining() {
    if (selectedDataFile.isEmpty()) {
        QMessageBox::warning(this, "No Data Selected", "Please select a training data file first.");
        return;
    }

    if (!QFile::exists(datasetPath)) {
        QMessageBox::warning(this, "Dataset Not Found", "Please preprocess the data first by selecting a data file.");
        return;
    }

    outputText->append("Starting training...");

    QStringList args;
    args << datasetPath;

    process->start("./lightwatch_train", args);
}

void MainWindow::startGeneration() {
    if (!QFile::exists("checkpoint_latest.bin")) {
        QMessageBox::warning(this, "No Checkpoint", "Please train a model first or ensure checkpoint_latest.bin exists.");
        return;
    }

    GenerationDialog dialog(this);
    if (dialog.exec() == QDialog::Accepted) {
        outputText->append("Starting generation...");

        QStringList args;
        args << "checkpoint_latest.bin";
        args << "--len" << QString::number(dialog.getLength());
        args << "--temp" << QString::number(dialog.getTemperature());
        args << "--topk" << QString::number(dialog.getTopK());
        if (!dialog.getSeed().isEmpty()) {
            args << "--seed" << dialog.getSeed();
        }

        process->start("./lightwatch_run", args);
    }
}

void MainWindow::readProcessOutput() {
    QByteArray output = process->readAllStandardOutput();
    QByteArray error = process->readAllStandardError();

    if (!output.isEmpty()) {
        outputText->append(QString::fromUtf8(output).trimmed());
    }
    if (!error.isEmpty()) {
        outputText->append(QString::fromUtf8(error).trimmed());
    }
}

void MainWindow::processFinished(int exitCode, QProcess::ExitStatus exitStatus) {
    if (exitStatus == QProcess::NormalExit) {
        outputText->append(QString("Process finished with exit code %1").arg(exitCode));
    } else {
        outputText->append("Process crashed or was killed");
    }
}

// GenerationDialog implementation
GenerationDialog::GenerationDialog(QWidget *parent) : QDialog(parent) {
    setWindowTitle("Generation Parameters");

    QFormLayout *layout = new QFormLayout(this);

    lengthSpin = new QSpinBox;
    lengthSpin->setRange(1, 10000);
    lengthSpin->setValue(100);
    layout->addRow("Length:", lengthSpin);

    tempSpin = new QDoubleSpinBox;
    tempSpin->setRange(0.1, 2.0);
    tempSpin->setSingleStep(0.1);
    tempSpin->setValue(1.0);
    layout->addRow("Temperature:", tempSpin);

    topKSpin = new QSpinBox;
    topKSpin->setRange(0, 1000);
    topKSpin->setValue(50);
    layout->addRow("Top-K:", topKSpin);

    seedEdit = new QLineEdit;
    layout->addRow("Seed:", seedEdit);

    QDialogButtonBox *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
    layout->addWidget(buttonBox);
}
