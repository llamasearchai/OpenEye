import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtMultimedia 5.15
import QtQml 2.15

ApplicationWindow {
    id: window
    visible: true
    width: 1280
    height: 720
    title: "OpenVideo - Drone Video Processing"
    
    // Main application state
    property var currentDetections: []
    property bool isProcessing: false
    property string rtspUrl: ""
    property var videoMetadata: ({})
    
    ColumnLayout {
        anchors.fill: parent
        spacing: 0
        
        // Top toolbar
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 60
            color: "#1a1a1a"
            
            RowLayout {
                anchors.fill: parent
                anchors.margins: 10
                spacing: 10
                
                Image {
                    Layout.preferredWidth: 40
                    Layout.preferredHeight: 40
                    source: "../assets/logo.png"
                    fillMode: Image.PreserveAspectFit
                }
                
                Label {
                    text: "OpenVideo"
                    font.pixelSize: 18
                    font.bold: true
                    color: "#ffffff"
                }
                
                Item { Layout.fillWidth: true }
                
                Button {
                    text: "Open Video"
                    onClicked: fileDialog.open()
                }
                
                Button {
                    text: isProcessing ? "Stop" : "Start Processing"
                    onClicked: {
                        if (isProcessing) {
                            stopProcessing();
                        } else {
                            startProcessing();
                        }
                    }
                }
                
                Button {
                    text: "AI Analysis"
                    onClicked: aiAnalysisDialog.open()
                }
            }
        }
        
        // Main content area
        SplitView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            orientation: Qt.Horizontal
            
            // Video player area
            Rectangle {
                SplitView.preferredWidth: parent.width * 0.7
                SplitView.minimumWidth: 400
                color: "#2a2a2a"
                
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 10
                    spacing: 10
                    
                    // Video display with overlay
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        color: "black"
                        
                        // Video player
                        VideoOutput {
                            id: videoOutput
                            anchors.fill: parent
                            source: mediaPlayer
                            fillMode: VideoOutput.PreserveAspectFit
                            
                            // Detection overlay
                            Canvas {
                                id: detectionOverlay
                                anchors.fill: parent
                                visible: true
                                
                                onPaint: {
                                    var ctx = getContext("2d");
                                    ctx.clearRect(0, 0, width, height);
                                    
                                    // Calculate scale factor based on video dimensions
                                    var videoWidth = videoMetadata.width || 1280;
                                    var videoHeight = videoMetadata.height || 720;
                                    var scaleX = width / videoWidth;
                                    var scaleY = height / videoHeight;
                                    
                                    // Draw detection boxes
                                    for (var i = 0; i < currentDetections.length; i++) {
                                        var det = currentDetections[i];
                                        var x = det.bbox[0] * scaleX;
                                        var y = det.bbox[1] * scaleY;
                                        var w = (det.bbox[2] - det.bbox[0]) * scaleX;
                                        var h = (det.bbox[3] - det.bbox[1]) * scaleY;
                                        
                                        // Draw bounding box
                                        ctx.strokeStyle = getClassColor(det.class_id);
                                        ctx.lineWidth = 2;
                                        ctx.strokeRect(x, y, w, h);
                                        
                                        // Draw label background
                                        ctx.fillStyle = getClassColor(det.class_id);
                                        var label = det.class_name + " (" + (det.confidence * 100).toFixed(1) + "%)";
                                        var labelWidth = ctx.measureText(label).width + 10;
                                        ctx.fillRect(x, y - 20, labelWidth, 20);
                                        
                                        // Draw label text
                                        ctx.fillStyle = "white";
                                        ctx.font = "12px Arial";
                                        ctx.fillText(label, x + 5, y - 5);
                                    }
                                }
                                
                                function getClassColor(classId) {
                                    // Color palette for different classes
                                    var colors = [
                                        "#FF5733", "#33FF57", "#3357FF", "#F3FF33", 
                                        "#FF33F3", "#33FFF3", "#F333FF", "#FF3F33"
                                    ];
                                    return colors[classId % colors.length];
                                }
                                
                                // Update overlay when detections change
                                Connections {
                                    target: window
                                    function onCurrentDetectionsChanged() {
                                        detectionOverlay.requestPaint();
                                    }
                                }
                            }
                        }
                        
                        MediaPlayer {
                            id: mediaPlayer
                            autoPlay: false
                            
                            onPositionChanged: {
                                // Request frame analysis at regular intervals when playing
                                if (playing && position % 500 === 0) {  // Every 500ms
                                    var frameIndex = Math.floor(position / 1000 * videoMetadata.fps);
                                    analyzeFrame(frameIndex);
                                }
                            }
                        }
                    }
                    
                    // Video controls
                    RowLayout {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 40
                        
                        Button {
                            text: mediaPlayer.playbackState === MediaPlayer.PlayingState ? "Pause" : "Play"
                            onClicked: {
                                if (mediaPlayer.playbackState === MediaPlayer.PlayingState)
                                    mediaPlayer.pause();
                                else
                                    mediaPlayer.play();
                            }
                        }
                        
                        Slider {
                            Layout.fillWidth: true
                            from: 0
                            to: mediaPlayer.duration
                            value: mediaPlayer.position
                            onMoved: mediaPlayer.seek(value)
                        }
                        
                        Label {
                            text: formatTime(mediaPlayer.position) + " / " + formatTime(mediaPlayer.duration)
                            color: "#ffffff"
                            
                            function formatTime(ms) {
                                var seconds = Math.floor(ms / 1000);
                                var minutes = Math.floor(seconds / 60);
                                seconds = seconds % 60;
                                return minutes + ":" + (seconds < 10 ? "0" : "") + seconds;
                            }
                        }
                    }
                }
            }
            
            // Sidebar with detection details and metrics
            Rectangle {
                SplitView.preferredWidth: parent.width * 0.3
                SplitView.minimumWidth: 250
                color: "#222222"
                
                TabBar {
                    id: sidebarTabs
                    anchors.top: parent.top
                    anchors.left: parent.left
                    anchors.right: parent.right
                    
                    TabButton {
                        text: "Detections"
                    }
                    
                    TabButton {
                        text: "Metrics"
                    }
                    
                    TabButton {
                        text: "Streams"
                    }
                }
                
                StackLayout {
                    anchors.top: sidebarTabs.bottom
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.bottom: parent.bottom
                    anchors.margins: 10
                    currentIndex: sidebarTabs.currentIndex
                    
                    // Detections panel
                    ColumnLayout {
                        spacing: 10
                        
                        Label {
                            text: "Current Detections"
                            font.pixelSize: 16
                            font.bold: true
                            color: "#ffffff"
                        }
                        
                        ScrollView {
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            clip: true
                            
                            ListView {
                                id: detectionsList
                                model: currentDetections
                                spacing: 5
                                
                                delegate: Rectangle {
                                    width: detectionsList.width
                                    height: 70
                                    color: "#333333"
                                    radius: 5
                                    
                                    ColumnLayout {
                                        anchors.fill: parent
                                        anchors.margins: 10
                                        spacing: 5
                                        
                                        Label {
                                            text: modelData.class_name
                                            font.pixelSize: 14
                                            font.bold: true
                                            color: "#ffffff"
                                        }
                                        
                                        Label {
                                            text: "Confidence: " + (modelData.confidence * 100).toFixed(1) + "%"
                                            color: "#cccccc"
                                        }
                                        
                                        Label {
                                            text: "Position: [" + 
                                                  modelData.bbox[0].toFixed(1) + ", " + 
                                                  modelData.bbox[1].toFixed(1) + ", " + 
                                                  modelData.bbox[2].toFixed(1) + ", " + 
                                                  modelData.bbox[3].toFixed(1) + "]"
                                            color: "#cccccc"
                                            font.pixelSize: 12
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    // Metrics panel
                    ColumnLayout {
                        spacing: 10
                        
                        Label {
                            text: "Performance Metrics"
                            font.pixelSize: 16
                            font.bold: true
                            color: "#ffffff"
                        }
                        
                        GridLayout {
                            columns: 2
                            Layout.fillWidth: true
                            columnSpacing: 10
                            rowSpacing: 10
                            
                            Label {
                                text: "FPS:"
                                color: "#cccccc"
                            }
                            
                            Label {
                                id: fpsMetric
                                text: "0"
                                color: "#ffffff"
                            }
                            
                            Label {
                                text: "Processing Time:"
                                color: "#cccccc"
                            }
                            
                            Label {
                                id: processingTimeMetric
                                text: "0 ms"
                                color: "#ffffff"
                            }
                            
                            Label {
                                text: "Detected Objects:"
                                color: "#cccccc"
                            }
                            
                            Label {
                                text: currentDetections.length.toString()
                                color: "#ffffff"
                            }
                            
                            Label {
                                text: "Video Resolution:"
                                color: "#cccccc"
                            }
                            
                            Label {
                                text: (videoMetadata.width || 0) + "x" + (videoMetadata.height || 0)
                                color: "#ffffff"
                            }
                        }
                        
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 150
                            color: "#333333"
                            radius: 5
                            
                            // Placeholder for performance graph
                            // In a real implementation, this would use a charting library
                            Canvas {
                                id: performanceGraph
                                anchors.fill: parent
                                anchors.margins: 10
                                
                                onPaint: {
                                    var ctx = getContext("2d");
                                    ctx.clearRect(0, 0, width, height);
                                    
                                    // Draw background grid
                                    ctx.strokeStyle = "#444444";
                                    ctx.lineWidth = 1;
                                    
                                    // Draw grid lines
                                    var gridSteps = 5;
                                    for (var i = 0; i <= gridSteps; i++) {
                                        var y = i * (height / gridSteps);
                                        ctx.beginPath();
                                        ctx.moveTo(0, y);
                                        ctx.lineTo(width, y);
                                        ctx.stroke();
                                    }
                                    
                                    // Placeholder for actual performance data
                                    ctx.strokeStyle = "#33FF57";
                                    ctx.lineWidth = 2;
                                    ctx.beginPath();
                                    ctx.moveTo(0, height);
                                    
                                    // Simulate some data points
                                    for (var x = 0; x < width; x += 10) {
                                        var y = height - Math.random() * height / 2 - height / 4;
                                        ctx.lineTo(x, y);
                                    }
                                    
                                    ctx.stroke();
                                }
                            }
                        }
                        
                        Label {
                            text: "CPU Usage"
                            color: "#cccccc"
                        }
                        
                        ProgressBar {
                            id: cpuUsageBar
                            Layout.fillWidth: true
                            value: 0.3 // Placeholder value
                        }
                        
                        Label {
                            text: "Memory Usage"
                            color: "#cccccc"
                        }
                        
                        ProgressBar {
                            id: memoryUsageBar
                            Layout.fillWidth: true
                            value: 0.5 // Placeholder value
                        }
                    }
                    
                    // Streams panel
                    ColumnLayout {
                        spacing: 10
                        
                        Label {
                            text: "Stream Endpoints"
                            font.pixelSize: 16
                            font.bold: true
                            color: "#ffffff"
                        }
                        
                        Rectangle {
                            Layout.fillWidth: true
                            height: 60
                            color: "#333333"
                            radius: 5
                            
                            RowLayout {
                                anchors.fill: parent
                                anchors.margins: 10
                                spacing: 10
                                
                                Label {
                                    text: "RTSP Stream:"
                                    color: "#cccccc"
                                }
                                
                                TextField {
                                    id: rtspUrlField
                                    Layout.fillWidth: true
                                    text: rtspUrl
                                    readOnly: true
                                    color: "#ffffff"
                                    background: Rectangle {
                                        color: "#444444"
                                        radius: 3
                                    }
                                }
                                
                                Button {
                                    text: "Copy"
                                    onClicked: {
                                        clipboardHelper.text = rtspUrlField.text;
                                        clipboardHelper.copy();
                                    }
                                }
                            }
                        }
                        
                        Rectangle {
                            Layout.fillWidth: true
                            height: 60
                            color: "#333333"
                            radius: 5
                            
                            RowLayout {
                                anchors.fill: parent
                                anchors.margins: 10
                                spacing: 10
                                
                                Label {
                                    text: "WebRTC Stream:"
                                    color: "#cccccc"
                                }
                                
                                TextField {
                                    Layout.fillWidth: true
                                    text: "https://localhost:8080/webrtc"
                                    readOnly: true
                                    color: "#ffffff"
                                    background: Rectangle {
                                        color: "#444444"
                                        radius: 3
                                    }
                                }
                                
                                Button {
                                    text: "Copy"
                                    onClicked: {
                                        // Copy to clipboard
                                    }
                                }
                            }
                        }
                        
                        Item {
                            Layout.fillHeight: true
                        }
                    }
                }
            }
        }
        
        // Status bar
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 30
            color: "#1a1a1a"
            
            RowLayout {
                anchors.fill: parent
                anchors.margins: 5
                spacing: 10
                
                Label {
                    text: "Status: " + (isProcessing ? "Processing" : "Ready")
                    color: "#cccccc"
                }
                
                Item { Layout.fillWidth: true }
                
                Label {
                    text: "OpenVideo v1.0"
                    color: "#cccccc"
                }
            }
        }
    }
    
    // AI Analysis Dialog
    Dialog {
        id: aiAnalysisDialog
        title: "AI Video Analysis"
        width: 600
        height: 500
        modal: true
        anchors.centerIn: parent
        
        background: Rectangle {
            color: "#222222"
            radius: 5
        }
        
        header: Rectangle {
            color: "#333333"
            height: 40
            
            Label {
                text: "AI Video Analysis"
                anchors.centerIn: parent
                font.pixelSize: 16
                font.bold: true
                color: "#ffffff"
            }
        }
        
        ColumnLayout {
            anchors.fill: parent
            anchors.margins: 20
            spacing: 20
            
            TabBar {
                id: aiTabs
                Layout.fillWidth: true
                
                TabButton {
                    text: "Natural Language Query"
                }
                
                TabButton {
                    text: "Video Summarization"
                }
            }
            
            StackLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                currentIndex: aiTabs.currentIndex
                
                // Natural Language Query Tab
                ColumnLayout {
                    spacing: 10
                    
                    Label {
                        text: "Ask a question about the video:"
                        color: "#ffffff"
                    }
                    
                    TextField {
                        id: queryField
                        Layout.fillWidth: true
                        placeholderText: "E.g., How many people are in the video?"
                        color: "#ffffff"
                        background: Rectangle {
                            color: "#333333"
                            radius: 3
                        }
                    }
                    
                    Button {
                        text: "Submit Query"
                        Layout.alignment: Qt.AlignRight
                        onClicked: {
                            queryResultArea.text = "Processing query...";
                            queryVideo(queryField.text);
                        }
                    }
                    
                    ScrollView {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        
                        TextArea {
                            id: queryResultArea
                            readOnly: true
                            wrapMode: TextEdit.Wrap
                            color: "#ffffff"
                            background: Rectangle {
                                color: "#333333"
                                radius: 3
                            }
                            textFormat: TextEdit.RichText
                        }
                    }
                }
                
                // Video Summarization Tab
                ColumnLayout {
                    spacing: 10
                    
                    Label {
                        text: "Generate an AI summary of the current video:"
                        color: "#ffffff"
                    }
                    
                    Button {
                        text: "Generate Summary"
                        Layout.alignment: Qt.AlignRight
                        onClicked: {
                            summaryResultArea.text = "Analyzing video...";
                            summarizeVideo();
                        }
                    }
                    
                    ScrollView {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        
                        TextArea {
                            id: summaryResultArea
                            readOnly: true
                            wrapMode: TextEdit.Wrap
                            color: "#ffffff"
                            background: Rectangle {
                                color: "#333333"
                                radius: 3
                            }
                        }
                    }
                }
            }
        }
        
        footer: DialogButtonBox {
            Button {
                text: "Close"
                DialogButtonBox.buttonRole: DialogButtonBox.RejectRole
            }
        }
    }
    
    // File dialog for opening videos
    FileDialog {
        id: fileDialog
        title: "Select Video File"
        nameFilters: ["Video files (*.mp4 *.avi *.mov *.mkv)"]
        onAccepted: {
            openVideo(fileDialog.fileUrl);
        }
    }
    
    // Clipboard helper
    TextEdit {
        id: clipboardHelper
        visible: false
        function copy() {
            selectAll();
            copySelectionToClipboard();
            deselect();
        }
    }
    
    // JavaScript functions to interact with Tauri
    function openVideo(path) {
        window.__TAURI__.invoke('open_video', { path: path.toString().replace(/^(file:\/{2})/,"") })
            .then(function(response) {
                if (response.success) {
                    videoMetadata = response.data;
                    mediaPlayer.source = path;
                    mediaPlayer.play();
                } else {
                    console.error("Error opening video:", response.error);
                }
            });
    }
    
    function startProcessing() {
        window.__TAURI__.invoke('start_processing')
            .then(function(response) {
                if (response.success) {
                    isProcessing = true;
                    
                    // Get RTSP URL after starting processing
                    getRtspUrl();
                    
                    // Start updating metrics
                    metricUpdateTimer.start();
                } else {
                    console.error("Error starting processing:", response.error);
                }
            });
    }
    
    function stopProcessing() {
        window.__TAURI__.invoke('stop_processing')
            .then(function(response) {
                if (response.success) {
                    isProcessing = false;
                    metricUpdateTimer.stop();
                } else {
                    console.error("Error stopping processing:", response.error);
                }
            });
    }
    
    function getRtspUrl() {
        window.__TAURI__.invoke('get_rtsp_url')
            .then(function(response) {
                if (response.success) {
                    rtspUrl = response.data.url;
                } else {
                    console.error("Error getting RTSP URL:", response.error);
                }
            });
    }
    
    function analyzeFrame(frameIndex) {
        window.__TAURI__.invoke('analyze_frame', { frame_index: frameIndex })
            .then(function(response) {
                if (response.success) {
                    currentDetections = response.data.detections;
                    
                    // Update processing metrics
                    if (response.data.processing_time) {
                        processingTimeMetric.text = response.data.processing_time.toFixed(1) + " ms";
                    }
                } else {
                    console.error("Error analyzing frame:", response.error);
                }
            });
    }
    
    function queryVideo(query) {
        window.__TAURI__.invoke('query_video', { query: query })
            .then(function(response) {
                if (response.success) {
                    var answer = response.data.answer;
                    var sources = response.data.sources || [];
                    
                    // Format the answer with sources
                    var formattedResult = "<p><b>Answer:</b> " + answer + "</p>";
                    
                    if (sources.length > 0) {
                        formattedResult += "<p><b>Sources:</b></p><ul>";
                        for (var i = 0; i < sources.length; i++) {
                            var source = sources[i];
                            var timestamp = new Date(source.timestamp * 1000).toISOString().substr(11, 8);
                            formattedResult += "<li>Video: " + source.video_id + " at " + timestamp + "</li>";
                        }
                        formattedResult += "</ul>";
                    }
                    
                    queryResultArea.text = formattedResult;
                } else {
                    queryResultArea.text = "Error: " + response.error;
                }
            });
    }
    
    function summarizeVideo() {
        if (!videoMetadata.path) {
            summaryResultArea.text = "Error: No video loaded";
            return;
        }
        
        window.__TAURI__.invoke('summarize_video', { video_path: videoMetadata.path })
            .then(function(response) {
                if (response.success) {
                    summaryResultArea.text = response.data.summary;
                } else {
                    summaryResultArea.text = "Error: " + response.error;
                }
            });
    }
    
    // Timer for updating metrics
    Timer {
        id: metricUpdateTimer
        interval: 1000
        repeat: true
        running: false
        
        onTriggered: {
            // In a real implementation, these would be retrieved from the backend
            fpsMetric.text = (15 + Math.random() * 10).toFixed(1);
            cpuUsageBar.value = Math.random() * 0.5 + 0.2;
            memoryUsageBar.value = Math.random() * 0.4 + 0.3;
            
            // Update the performance graph
            performanceGraph.requestPaint();
        }
    }
    
    // Initial setup
    Component.onCompleted: {
        // Initialize any required components
        getRtspUrl();
    }
}