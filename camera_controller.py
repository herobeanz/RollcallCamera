"""
ONVIF Camera Controller
Handles connection, authentication, and RTSP URL retrieval from IP cameras via ONVIF protocol.
"""
import logging
from typing import Optional, List, Dict, Tuple
from onvif import ONVIFCamera
from zeep.exceptions import Fault


class CameraController:
    """
    Controller for ONVIF-compatible IP cameras.
    Handles authentication, media profile queries, and RTSP stream URL retrieval.
    """
    
    def __init__(self, host: str, port: int = 80, username: str = "admin", password: str = "password"):
        """
        Initialize ONVIF camera controller.
        
        Args:
            host: Camera IP address or hostname
            port: ONVIF service port (typically 80, 8080, or 554)
            username: Camera username for authentication
            password: Camera password for authentication
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.camera: Optional[ONVIFCamera] = None
        self.media_service = None
        self.ptz_service = None
        self.profiles: List[Dict] = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def connect(self) -> bool:
        """
        Connect to the ONVIF camera and initialize services.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.logger.info(f"Connecting to ONVIF camera at {self.host}:{self.port}")
            
            # Create ONVIF camera instance
            self.camera = ONVIFCamera(
                self.host,
                self.port,
                self.username,
                self.password
            )
            
            # Get media service (required for stream URLs)
            self.media_service = self.camera.create_media_service()
            self.logger.info("Media service initialized")
            
            # Try to get PTZ service (optional, may not be available)
            try:
                self.ptz_service = self.camera.create_ptz_service()
                self.logger.info("PTZ service initialized")
            except Exception as e:
                self.logger.warning(f"PTZ service not available: {str(e)}")
                self.ptz_service = None
            
            # Query available profiles
            self._query_profiles()
            
            self.logger.info(f"Successfully connected to camera at {self.host}:{self.port}")
            return True
            
        except Fault as e:
            self.logger.error(f"ONVIF Fault error: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Connection error: {str(e)}")
            return False
    
    def _query_profiles(self) -> None:
        """
        Query and cache available media profiles from the camera.
        """
        try:
            profiles = self.media_service.GetProfiles()
            self.profiles = []
            
            for profile in profiles:
                profile_info = {
                    'token': profile.token,
                    'name': profile.Name if hasattr(profile, 'Name') else profile.token,
                    'video_encoder': None,
                    'audio_encoder': None,
                    'video_resolution': None
                }
                
                # Extract video encoder configuration
                if hasattr(profile, 'VideoEncoderConfiguration'):
                    venc = profile.VideoEncoderConfiguration
                    profile_info['video_encoder'] = {
                        'token': venc.token if hasattr(venc, 'token') else None,
                        'resolution': None
                    }
                    if hasattr(venc, 'Resolution'):
                        res = venc.Resolution
                        profile_info['video_resolution'] = {
                            'width': res.Width if hasattr(res, 'Width') else None,
                            'height': res.Height if hasattr(res, 'Height') else None
                        }
                        profile_info['video_encoder']['resolution'] = profile_info['video_resolution']
                
                # Extract audio encoder configuration
                if hasattr(profile, 'AudioEncoderConfiguration'):
                    aenc = profile.AudioEncoderConfiguration
                    profile_info['audio_encoder'] = {
                        'token': aenc.token if hasattr(aenc, 'token') else None
                    }
                
                self.profiles.append(profile_info)
            
            self.logger.info(f"Found {len(self.profiles)} media profile(s)")
            for i, profile in enumerate(self.profiles):
                res_str = "N/A"
                if profile['video_resolution']:
                    w = profile['video_resolution']['width']
                    h = profile['video_resolution']['height']
                    res_str = f"{w}x{h}" if w and h else "N/A"
                self.logger.info(f"  Profile {i+1}: {profile['name']} ({res_str})")
                
        except Exception as e:
            self.logger.error(f"Error querying profiles: {str(e)}")
            raise
    
    def get_profiles(self) -> List[Dict]:
        """
        Get list of available media profiles.
        
        Returns:
            List of profile dictionaries with token, name, and configuration info
        """
        return self.profiles
    
    def get_rtsp_url(self, profile_token: Optional[str] = None, transport: str = "RTP-Unicast") -> Optional[str]:
        """
        Get RTSP stream URL for a specific profile.
        
        Args:
            profile_token: Profile token (uses first profile if None)
            transport: Transport protocol ("RTP-Unicast" or "RTP-Multicast")
            
        Returns:
            RTSP URL string, or None if not available
        """
        if not self.media_service:
            self.logger.error("Media service not initialized. Call connect() first.")
            return None
        
        try:
            # Use first profile if not specified
            if profile_token is None:
                if not self.profiles:
                    self.logger.error("No profiles available")
                    return None
                profile_token = self.profiles[0]['token']
            
            # Call GetStreamUri directly with dictionary parameters
            # This avoids the StreamSetup type creation issue
            # Some cameras work better with direct parameter passing
            try:
                stream_uri = self.media_service.GetStreamUri({
                    'ProfileToken': profile_token,
                    'StreamSetup': {
                        'Stream': 'RTP-Unicast',
                        'Transport': {'Protocol': 'RTSP'}
                    }
                })
            except Exception as e1:
                # If dictionary format fails, try with create_type but simpler
                self.logger.debug(f"Direct call failed: {str(e1)}, trying create_type method...")
                uri = self.media_service.create_type('GetStreamUri')
                uri.ProfileToken = profile_token
                
                # Try to get StreamSetup type from the underlying client
                try:
                    # Access the zeep client directly
                    client = self.media_service._binding._default_service._binding._client
                    # Get StreamSetup type from schema
                    StreamSetupType = client.get_type('ns0:StreamSetup')
                    stream_setup = StreamSetupType(Stream='RTP-Unicast', Transport={'Protocol': 'RTSP'})
                    uri.StreamSetup = stream_setup
                    stream_uri = self.media_service.GetStreamUri(uri)
                except Exception as e2:
                    self.logger.error(f"Both methods failed. Last error: {str(e2)}")
                    raise
            rtsp_url = stream_uri.Uri if hasattr(stream_uri, 'Uri') else None
            
            if rtsp_url:
                # Replace username/password in URL if needed
                # Some cameras return URLs without credentials
                if '://' in rtsp_url and '@' not in rtsp_url:
                    # Insert credentials into URL
                    protocol_end = rtsp_url.find('://') + 3
                    rtsp_url = f"{rtsp_url[:protocol_end]}{self.username}:{self.password}@{rtsp_url[protocol_end:]}"
                
                self.logger.info(f"Retrieved RTSP URL for profile '{profile_token}': {rtsp_url[:50]}...")
                return rtsp_url
            else:
                self.logger.error("Stream URI is empty")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting RTSP URL: {str(e)}")
            return None
    
    def get_rtsp_urls(self) -> Dict[str, str]:
        """
        Get RTSP URLs for all available profiles.
        
        Returns:
            Dictionary mapping profile tokens to RTSP URLs
        """
        urls = {}
        for profile in self.profiles:
            token = profile['token']
            url = self.get_rtsp_url(token)
            if url:
                urls[token] = url
        return urls
    
    # ==================== PTZ Control Functions (Optional) ====================
    
    def is_ptz_supported(self) -> bool:
        """Check if PTZ control is supported."""
        return self.ptz_service is not None
    
    def get_ptz_status(self) -> Optional[Dict]:
        """
        Get current PTZ status (position, zoom, etc.).
        
        Returns:
            Dictionary with PTZ status or None if not supported
        """
        if not self.ptz_service:
            self.logger.warning("PTZ service not available")
            return None
        
        try:
            # Get PTZ configuration
            ptz_config = self.ptz_service.GetConfigurationOptions({'ConfigurationToken': self.profiles[0]['token']})
            
            # Get current status
            status = self.ptz_service.GetStatus({'ProfileToken': self.profiles[0]['token']})
            
            return {
                'position': {
                    'pan': status.Position.PanTilt.x if hasattr(status, 'Position') else None,
                    'tilt': status.Position.PanTilt.y if hasattr(status, 'Position') else None,
                    'zoom': status.Position.Zoom.x if hasattr(status, 'Position') else None
                },
                'move_status': {
                    'pan_tilt': status.MoveStatus.PanTilt if hasattr(status, 'MoveStatus') else None,
                    'zoom': status.MoveStatus.Zoom if hasattr(status, 'MoveStatus') else None
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting PTZ status: {str(e)}")
            return None
    
    def move_ptz(self, pan: float = 0.0, tilt: float = 0.0, zoom: float = 0.0, 
                 profile_token: Optional[str] = None, speed: float = 0.5) -> bool:
        """
        Move PTZ camera (pan, tilt, zoom).
        
        Args:
            pan: Pan value (-1.0 to 1.0)
            tilt: Tilt value (-1.0 to 1.0)
            zoom: Zoom value (0.0 to 1.0)
            profile_token: Profile token (uses first if None)
            speed: Movement speed (0.0 to 1.0)
            
        Returns:
            True if command sent successfully
        """
        if not self.ptz_service:
            self.logger.warning("PTZ service not available")
            return False
        
        try:
            if profile_token is None:
                profile_token = self.profiles[0]['token']
            
            # Create continuous move request
            move_request = self.ptz_service.create_type('ContinuousMove')
            move_request.ProfileToken = profile_token
            move_request.Velocity = {
                'PanTilt': {'x': pan * speed, 'y': tilt * speed},
                'Zoom': {'x': zoom * speed}
            }
            
            self.ptz_service.ContinuousMove(move_request)
            self.logger.info(f"PTZ move command: pan={pan}, tilt={tilt}, zoom={zoom}, speed={speed}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error moving PTZ: {str(e)}")
            return False
    
    def stop_ptz(self, profile_token: Optional[str] = None) -> bool:
        """
        Stop PTZ movement.
        
        Args:
            profile_token: Profile token (uses first if None)
            
        Returns:
            True if command sent successfully
        """
        if not self.ptz_service:
            return False
        
        try:
            if profile_token is None:
                profile_token = self.profiles[0]['token']
            
            stop_request = self.ptz_service.create_type('Stop')
            stop_request.ProfileToken = profile_token
            stop_request.PanTilt = True
            stop_request.Zoom = True
            
            self.ptz_service.Stop(stop_request)
            self.logger.info("PTZ stop command sent")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping PTZ: {str(e)}")
            return False
    
    def disconnect(self) -> None:
        """Clean up resources."""
        self.camera = None
        self.media_service = None
        self.ptz_service = None
        self.profiles = []
        self.logger.info("Disconnected from camera")
