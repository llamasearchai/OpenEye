import React, { useEffect, useRef, useState } from 'react';

const WebRTCPlayer = ({ apiUrl = 'http://localhost:8000/webrtc' }) => {
  const videoRef = useRef(null);
  const [status, setStatus] = useState('Initializing...');
  const [peerConnection, setPeerConnection] = useState(null);
  const [connectionId, setConnectionId] = useState(null);

  useEffect(() => {
    let pc = null;
    
    const connectWebRTC = async () => {
      try {
        setStatus('Connecting...');
        
        // Create peer connection with STUN servers
        pc = new RTCPeerConnection({
          iceServers: [
            {
              urls: 'stun:stun.l.google.com:19302',
            },
          ],
        });
        
        setPeerConnection(pc);
        
        // Set up event handlers
        pc.ontrack = (event) => {
          if (videoRef.current && event.streams[0]) {
            videoRef.current.srcObject = event.streams[0];
            setStatus('Connected');
          }
        };
        
        pc.onicecandidate = (event) => {
          if (event.candidate) {
            // Send ICE candidate to server
            sendIceCandidate(event.candidate, connectionId);
          }
        };
        
        pc.oniceconnectionstatechange = () => {
          setStatus(`ICE Connection: ${pc.iceConnectionState}`);
          
          if (pc.iceConnectionState === 'disconnected' || 
              pc.iceConnectionState === 'failed' || 
              pc.iceConnectionState === 'closed') {
            setStatus('Disconnected');
            // Attempt reconnect after a delay
            setTimeout(() => {
              if (videoRef.current) {
                connectWebRTC();
              }
            }, 5000);
          }
        };
        
        // Get SDP offer from server
        const response = await fetch(`${apiUrl}/offer`);
        if (!response.ok) {
          throw new Error(`Server returned ${response.status}: ${await response.text()}`);
        }
        
        const { sdp, type, pc_id } = await response.json();
        setConnectionId(pc_id);
        
        // Set remote description (server's offer)
        await pc.setRemoteDescription(new RTCSessionDescription({ sdp, type }));
        
        // Create answer
        const answer = await pc.createAnswer();
        await pc.setLocalDescription(answer);
        
        // Send answer to server
        await fetch(`${apiUrl}/answer`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            sdp: pc.localDescription.sdp,
            type: pc.localDescription.type,
            pc_id: pc_id,
          }),
        });
        
        setStatus('Connected - Waiting for video');
        
      } catch (err) {
        console.error('Error connecting to WebRTC:', err);
        setStatus(`Error: ${err.message}`);
        
        // Clean up on error
        if (pc) {
          pc.close();
        }
        
        // Attempt reconnect after a delay
        setTimeout(() => {
          if (videoRef.current) {
            connectWebRTC();
          }
        }, 5000);
      }
    };
    
    const sendIceCandidate = async (candidate, pcId) => {
      if (!pcId) return;
      
      try {
        await fetch(`${apiUrl}/ice_candidate`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            candidate: candidate,
            pc_id: pcId,
          }),
        });
      } catch (err) {
        console.error('Error sending ICE candidate:', err);
      }
    };
    
    // Connect when component mounts
    connectWebRTC();
    
    // Clean up when component unmounts
    return () => {
      setStatus('Disconnected');
      if (peerConnection) {
        peerConnection.close();
      }
    };
  }, [apiUrl]);
  
  return (
    <div className="webrtc-player">
      <div className="status-bar">{status}</div>
      <video 
        ref={videoRef} 
        autoPlay 
        playsInline 
        muted 
        style={{ width: '100%', maxHeight: '80vh' }}
      />
    </div>
  );
};

export default WebRTCPlayer;