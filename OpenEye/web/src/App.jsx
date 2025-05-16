import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Button, Form, Spinner } from 'react-bootstrap';
import WebRTCPlayer from './components/WebRTCPlayer';
import ApiService from './services/ApiService';
import './App.css';

const App = () => {
  const [apiService] = useState(new ApiService());
  const [videoPath, setVideoPath] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [videoMetadata, setVideoMetadata] = useState(null);
  const [query, setQuery] = useState('');
  const [queryResult, setQueryResult] = useState(null);
  const [rtspUrl, setRtspUrl] = useState('');
  const [currentFrame, setCurrentFrame] = useState(0);
  const [frameAnalysis, setFrameAnalysis] = useState(null);
  
  const handleOpenVideo = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    
    try {
      const result = await apiService.openVideo(videoPath);
      if (result.success) {
        setIsConnected(true);
        setVideoMetadata(result.data.metadata);
      } else {
        alert(`Failed to open video: ${result.error}`);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleQuery = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    
    try {
      const result = await apiService.queryVideo(query);
      setQueryResult(result.data);
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleStartRtsp = async () => {
    setIsLoading(true);
    
    try {
      const result = await apiService.startRtspServer();
      if (result.data.success) {
        setRtspUrl(result.data.url);
      } else {
        alert(`Failed to start RTSP server: ${result.error}`);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleAnalyzeFrame = async () => {
    setIsLoading(true);
    
    try {
      const result = await apiService.analyzeFrame(currentFrame);
      if (result.data.success) {
        setFrameAnalysis(result.data.analysis);
      } else {
        alert(`Failed to analyze frame: ${result.data.error}`);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <Container fluid className="mt-4">
      <Row>
        <Col>
          <h1>OpenVideo</h1>
          <p className="lead">Real-time video processing and analysis for drone camera systems</p>
        </Col>
      </Row>
      
      {!isConnected ? (
        <Row className="mt-3">
          <Col md={6}>
            <Card>
              <Card.Header>Connect to Video Source</Card.Header>
              <Card.Body>
                <Form onSubmit={handleOpenVideo}>
                  <Form.Group>
                    <Form.Label>Video Path, Camera Index, or RTSP URL</Form.Label>
                    <Form.Control
                      type="text"
                      value={videoPath}
                      onChange={(e) => setVideoPath(e.target.value)}
                      placeholder="0 for webcam, file path, or rtsp://..."
                    />
                  </Form.Group>
                  <Button 
                    type="submit" 
                    variant="primary" 
                    className="mt-3" 
                    disabled={isLoading}
                  >
                    {isLoading ? <Spinner animation="border" size="sm" /> : 'Connect'}
                  </Button>
                </Form>
              </Card.Body>
            </Card>
          </Col>
        </Row>
      ) : (
        <>
          <Row className="mt-3">
            <Col md={8}>
              <Card>
                <Card.Header>
                  Video Stream
                  {videoMetadata && (
                    <span className="ms-2 text-muted">
                      {videoMetadata.width}x{videoMetadata.height} @ {videoMetadata.fps}fps
                    </span>
                  )}
                </Card.Header>
                <Card.Body>
                  <WebRTCPlayer />
                </Card.Body>
              </Card>
            </Col>
            
            <Col md={4}>
              <Card>
                <Card.Header>Video Controls</Card.Header>
                <Card.Body>
                  <div className="mb-3">
                    <Button onClick={handleStartRtsp} disabled={rtspUrl || isLoading}>
                      {isLoading ? <Spinner animation="border" size="sm" /> : 'Start RTSP Server'}
                    </Button>
                    
                    {rtspUrl && (
                      <div className="mt-2">
                        <small>RTSP URL: <code>{rtspUrl}</code></small>
                      </div>
                    )}
                  </div>
                  
                  <div className="mb-3">
                    <Form.Label>Current Frame</Form.Label>
                    <div className="d-flex">
                      <Form.Control
                        type="number"
                        value={currentFrame}
                        onChange={(e) => setCurrentFrame(parseInt(e.target.value))}
                        min="0"
                      />
                      <Button 
                        variant="secondary" 
                        className="ms-2" 
                        onClick={handleAnalyzeFrame}
                        disabled={isLoading}
                      >
                        Analyze
                      </Button>
                    </div>
                  </div>
                </Card.Body>
              </Card>
              
              <Card className="mt-3">
                <Card.Header>Natural Language Query</Card.Header>
                <Card.Body>
                  <Form onSubmit={handleQuery}>
                    <Form.Group>
                      <Form.Control
                        as="textarea"
                        rows={2}
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Ask a question about the video..."
                      />
                    </Form.Group>
                    <Button 
                      type="submit" 
                      variant="primary" 
                      className="mt-2" 
                      disabled={isLoading}
                    >
                      {isLoading ? <Spinner animation="border" size="sm" /> : 'Query'}
                    </Button>
                  </Form>
                  
                  {queryResult && (
                    <div className="mt-3">
                      <h6>Answer:</h6>
                      <p>{queryResult.answer}</p>
                      
                      {queryResult.sources && queryResult.sources.length > 0 && (
                        <>
                          <h6>Sources:</h6>
                          <ul className="small">
                            {queryResult.sources.map((source, idx) => (
                              <li key={idx}>
                                {source.source_type}: {source.content}
                                {source.timestamp && (
                                  <span className="ms-1 text-muted">
                                    at {new Date(source.timestamp * 1000).toISOString().substr(11, 8)}
                                  </span>
                                )}
                              </li>
                            ))}
                          </ul>
                        </>
                      )}
                    </div>
                  )}
                </Card.Body>
              </Card>
            </Col>
          </Row>
          
          {frameAnalysis && (
            <Row className="mt-3">
              <Col>
                <Card>
                  <Card.Header>Frame Analysis</Card.Header>
                  <Card.Body>
                    <Row>
                      <Col md={4}>
                        {frameAnalysis.frame_base64 && (
                          <img 
                            src={`data:image/jpeg;base64,${frameAnalysis.frame_base64}`}
                            alt="Analyzed frame"
                            className="img-fluid"
                          />
                        )}
                      </Col>
                      <Col md={8}>
                        <h5>AI Analysis:</h5>
                        <p>{frameAnalysis.result}</p>
                        
                        {frameAnalysis.detections && frameAnalysis.detections.length > 0 && (
                          <>
                            <h5>Detections:</h5>
                            <ul>
                              {frameAnalysis.detections.map((detection, idx) => (
                                <li key={idx}>
                                  {detection.class_name} ({(detection.confidence * 100).toFixed(1)}%)
                                  at [{detection.bbox.join(', ')}]
                                </li>
                              ))}
                            </ul>
                          </>
                        )}
                      </Col>
                    </Row>
                  </Card.Body>
                </Card>
              </Col>
            </Row>
          )}
        </>
      )}
      
      <footer className="mt-5 mb-3 text-center text-muted">
        <small>OpenVideo - Comprehensive video processing for drone camera systems</small>
      </footer>
    </Container>
  );
};

export default App;