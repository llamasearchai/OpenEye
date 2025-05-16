import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Rectangle {
    id: root
    color: "#222222"
    
    property var telemetryData: null
    property var detections: []
    
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 10
        
        Label {
            text: "Drone Analytics"
            font.pixelSize: 18
            font.bold: true
            color: "#ffffff"
        }
        
        // Telemetry display
        GroupBox {
            title: "Telemetry Data"
            Layout.fillWidth: true
            
            background: Rectangle {
                color: "#333333"
                border.color: "#555555"
                radius: 5
            }
            
            label: Label {
                text: parent.title
                color: "#ffffff"
                padding: 5
            }
            
            GridLayout {
                columns: 2
                rowSpacing: 5
                columnSpacing: 10
                
                Label { text: "Altitude:"; color: "#cccccc" }
                Label { 
                    text: telemetryData ? telemetryData.altitude.toFixed(1) + " m" : "N/A"
                    color: "#ffffff" 
                }
                
                Label { text: "GPS:"; color: "#cccccc" }
                Label { 
                    text: telemetryData ? 
                          telemetryData.latitude.toFixed(6) + ", " + 
                          telemetryData.longitude.toFixed(6) : "N/A"
                    color: "#ffffff" 
                }
                
                Label { text: "Speed:"; color: "#cccccc" }
                Label { 
                    text: telemetryData ? telemetryData.speed.toFixed(1) + " m/s" : "N/A"
                    color: "#ffffff" 
                }
                
                Label { text: "Heading:"; color: "#cccccc" }
                Label { 
                    text: telemetryData ? telemetryData.heading.toFixed(1) + "°" : "N/A"
                    color: "#ffffff" 
                }
            }
        }
        
        // Aerial intelligence
        GroupBox {
            title: "Aerial Intelligence"
            Layout.fillWidth: true
            Layout.fillHeight: true
            
            background: Rectangle {
                color: "#333333"
                border.color: "#555555"
                radius: 5
            }
            
            label: Label {
                text: parent.title
                color: "#ffffff"
                padding: 5
            }
            
            ColumnLayout {
                anchors.fill: parent
                spacing: 10
                
                // Object count summary
                Rectangle {
                    Layout.fillWidth: true
                    height: 40
                    color: "#444444"
                    radius: 3
                    
                    Label {
                        anchors.fill: parent
                        anchors.margins: 10
                        text: getObjectSummary()
                        color: "#ffffff"
                    }
                    
                    function getObjectSummary() {
                        if (!detections || detections.length === 0)
                            return "No objects detected";
                            
                        var counts = {};
                        for (var i = 0; i < detections.length; i++) {
                            var className = detections[i].class_name;
                            counts[className] = (counts[className] || 0) + 1;
                        }
                        
                        var result = "Detected: ";
                        var items = [];
                        for (var key in counts) {
                            items.push(counts[key] + " " + key);
                        }
                        
                        return result + items.join(", ");
                    }
                }
                
                // Map visualization (placeholder)
                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    color: "#2a2a2a"
                    
                    Canvas {
                        id: mapCanvas
                        anchors.fill: parent
                        
                        onPaint: {
                            var ctx = getContext("2d");
                            ctx.fillStyle = "#2a2a2a";
                            ctx.fillRect(0, 0, width, height);
                            
                            // Draw grid
                            ctx.strokeStyle = "#444444";
                            ctx.lineWidth = 1;
                            
                            var gridSize = 30;
                            for (var x = 0; x < width; x += gridSize) {
                                ctx.beginPath();
                                ctx.moveTo(x, 0);
                                ctx.lineTo(x, height);
                                ctx.stroke();
                            }
                            
                            for (var y = 0; y < height; y += gridSize) {
                                ctx.beginPath();
                                ctx.moveTo(0, y);
                                ctx.lineTo(width, y);
                                ctx.stroke();
                            }
                            
                            // Draw drone position (center)
                            var centerX = width / 2;
                            var centerY = height / 2;
                            
                            ctx.fillStyle = "#33AAFF";
                            ctx.beginPath();
                            ctx.arc(centerX, centerY, 8, 0, Math.PI * 2);
                            ctx.fill();
                            
                            // Draw drone direction
                            if (telemetryData) {
                                var heading = telemetryData.heading * Math.PI / 180;
                                var dirX = centerX + Math.sin(heading) * 20;
                                var dirY = centerY - Math.cos(heading) * 20;
                                
                                ctx.strokeStyle = "#33AAFF";
                                ctx.lineWidth = 2;
                                ctx.beginPath();
                                ctx.moveTo(centerX, centerY);
                                ctx.lineTo(dirX, dirY);
                                ctx.stroke();
                            }
                            
                            // Draw detections as points on the map
                            if (detections) {
                                for (var i = 0; i < detections.length; i++) {
                                    var det = detections[i];
                                    
                                    // Calculate detection position relative to drone
                                    // This is a simplified example - in reality would use proper coordinate transforms
                                    var bbox = det.bbox;
                                    var detCenterX = (bbox[0] + bbox[2]) / 2;
                                    var detCenterY = (bbox[1] + bbox[3]) / 2;
                                    
                                    // Map to canvas coordinates (simplified)
                                    var mapX = centerX + (detCenterX - 0.5) * width * 0.8;
                                    var mapY = centerY + (detCenterY - 0.5) * height * 0.8;
                                    
                                    // Draw detection point
                                    ctx.fillStyle = getClassColor(det.class_id);
                                    ctx.beginPath();
                                    ctx.arc(mapX, mapY, 5, 0, Math.PI * 2);
                                    ctx.fill();
                                }
                            }
                        }
                        
                        function getClassColor(classId) {
                            var colors = [
                                "#FF5733", "#33FF57", "#3357FF", "#F3FF33", 
                                "#FF33F3", "#33FFF3", "#F333FF", "#FF3F33"
                            ];
                            return colors[classId % colors.length];
                        }
                    }
                    
                    // Update map when telemetry or detections change
                    Connections {
                        target: root
                        function onTelemetryDataChanged() { mapCanvas.requestPaint(); }
                        function onDetectionsChanged() { mapCanvas.requestPaint(); }
                    }
                }
            }
        }
        
        // KLV Metadata
        GroupBox {
            title: "KLV Metadata"
            Layout.fillWidth: true
            
            background: Rectangle {
                color: "#333333"
                border.color: "#555555"
                radius: 5
            }
            
            label: Label {
                text: parent.title
                color: "#ffffff"
                padding: 5
            }
            
            ColumnLayout {
                anchors.fill: parent
                spacing: 5
                
                Label {
                    text: "Embedded MISB KLV metadata extracted from stream:"
                    color: "#cccccc"
                    font.pixelSize: 12
                }
                
                ScrollView {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 80
                    clip: true
                    
                    TextArea {
                        readOnly: true
                        text: formatKlvMetadata()
                        color: "#ffffff"
                        wrapMode: TextEdit.Wrap
                        background: Rectangle {
                            color: "#444444"
                            radius: 3
                        }
                    }
                    
                    function formatKlvMetadata() {
                        if (!telemetryData || !telemetryData.klv)
                            return "No KLV metadata available";
                            
                        var klv = telemetryData.klv;
                        var result = "";
                        
                        for (var key in klv) {
                            result += key + ": " + klv[key] + "\n";
                        }
                        
                        return result;
                    }
                }
            }
        }
    }
}