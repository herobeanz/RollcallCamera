"""
Script chính để chạy nhận diện khuôn mặt và điểm danh từ IP camera real-time
Tích hợp toàn bộ logic từ video tĩnh vào camera streaming
"""
import argparse
import json
import os
import sys
import cv2
import logging
from datetime import datetime
from typing import TYPE_CHECKING
import config
from camera_controller import CameraController

# Type hints chỉ để kiểm tra type, không import thực sự
if TYPE_CHECKING:
    from video_processor import VideoProcessor
    from employee_manager import EmployeeManager
else:
    VideoProcessor = None
    EmployeeManager = None

# Các import chỉ cần khi chạy chế độ nhận diện (không cần cho stream-only)
# Sẽ import khi cần trong các hàm tương ứng


def setup_logging(level: int = logging.INFO, log_file: str = None) -> None:
    """
    Setup logging configuration with file and console handlers.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (optional, uses config if None)
    """
    import logging.handlers
    
    if log_file is None:
        log_file = getattr(config, 'LOG_FILE', os.path.join(getattr(config, 'LOGS_DIR', 'logs'), 'kitchen_cam.log'))
    
    # Tạo thư mục log nếu chưa có
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Tạo formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Xóa handlers cũ nếu có
    root_logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (nếu ENABLE_LOGGING)
    if getattr(config, 'ENABLE_LOGGING', True):
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def validate_file_path(path: str, file_type: str = "file") -> bool:
    """
    Validate file or directory path exists.
    
    Args:
        path: Path to validate
        file_type: Type of path ("file" or "directory")
        
    Returns:
        True if valid, raises error otherwise
    """
    if not path:
        raise ValueError(f"{file_type.capitalize()} path is required")
    
    if file_type == "file":
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        if not os.path.isfile(path):
            raise ValueError(f"Path is not a file: {path}")
    else:  # directory
        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory not found: {path}")
        if not os.path.isdir(path):
            raise ValueError(f"Path is not a directory: {path}")
    
    return True


def validate_json_file(file_path: str) -> bool:
    """
    Validate JSON file format.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        True if valid, raises error otherwise
    """
    validate_file_path(file_path, "file")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
        return True
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {file_path}: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error reading JSON file {file_path}: {str(e)}")


def load_registered_employees(registered_file: str) -> list:
    """
    Load danh sách nhân viên đã đăng ký từ file JSON
    
    Args:
        registered_file: Đường dẫn file JSON
        
    Returns:
        List employee_id
    """
    try:
        validate_json_file(registered_file)
    except (FileNotFoundError, ValueError) as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Registered file validation failed: {str(e)}")
        return []
    
    try:
        with open(registered_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ho tro format: {"registered_employees": ["001", "002", ...]}
        # hoac {"date": "...", "registered_employees": [...]}
        if isinstance(data, dict):
            employees = data.get('registered_employees', [])
        elif isinstance(data, list):
            employees = data
        else:
            employees = []
        
        print(f"Da load {len(employees)} nhan vien dang ky tu: {registered_file}")
        return employees
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"ERROR: Loi khi doc file dang ky: {str(e)}")
        return []


def stream_simple(controller: CameraController, window_name: str = "Camera Stream", exit_key: str = 'q'):
    """
    Hiển thị video stream đơn giản từ camera (chưa có nhận diện khuôn mặt)
    
    Args:
        controller: CameraController instance
        window_name: Tên cửa sổ hiển thị
        exit_key: Phím để thoát
    """
    logger = logging.getLogger(__name__)
    
    # Lấy RTSP URL
    logger.info("Đang lấy RTSP URL...")
    rtsp_url = controller.get_rtsp_url()
    
    if not rtsp_url:
        logger.error("Không thể lấy RTSP URL!")
        return False
    
    logger.info(f"RTSP URL: {rtsp_url[:80]}...")
    
    logger.info("Đang mở video stream...")
    
    # Mở video stream
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        logger.error("Không thể mở video stream!")
        logger.error("Vui lòng kiểm tra:")
        logger.error("  - RTSP URL có đúng không")
        logger.error("  - Camera có đang stream không")
        logger.error("  - Firewall có chặn RTSP không")
        return False
    
    # Lấy thông tin stream
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"Stream: {width}x{height} @ {fps:.2f} FPS")
    logger.info(f"Nhấn '{exit_key}' để thoát")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret or frame is None:
                logger.warning("Không thể đọc frame, đang thử lại...")
                continue
            
            frame_count += 1
            
            # Vẽ thông tin trên frame
            info_text = f"Frame: {frame_count} | {width}x{height} @ {fps:.1f} FPS"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Vẽ hướng dẫn
            help_text = "Press 'q' to quit"
            cv2.putText(frame, help_text, (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Hiển thị frame
            cv2.imshow(window_name, frame)
            
            # Kiểm tra phím 'q' để thoát
            key = cv2.waitKey(1) & 0xFF
            if key == ord(exit_key):
                logger.info("Người dùng nhấn 'q' để thoát")
                break
            
    except KeyboardInterrupt:
        logger.info("Bị ngắt bởi người dùng (Ctrl+C)")
    except Exception as e:
        logger.error(f"Lỗi khi hiển thị stream: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Đã đóng video stream")
    
    return True


def stream_with_attendance(controller: CameraController, video_processor: VideoProcessor,
                           window_name: str = "Camera Attendance", exit_key: str = 'q',
                           frame_skip: int = 0, auto_optimize: bool = True,
                           low_latency: bool = True, drop_old_frames: bool = True):
    """
    Stream video từ camera và xử lý nhận diện khuôn mặt real-time
    
    Args:
        controller: CameraController instance
        video_processor: VideoProcessor instance
        window_name: Tên cửa sổ hiển thị
        exit_key: Phím để thoát
        frame_skip: Số frame bỏ qua giữa 2 lần xử lý
    """
    from video_stream import VideoStream
    from visualization import draw_statistics
    
    logger = logging.getLogger(__name__)
    
    # Lấy RTSP URL
    logger.info("Retrieving RTSP stream URL...")
    rtsp_url = controller.get_rtsp_url()
    
    if not rtsp_url:
        logger.error("Failed to retrieve RTSP URL")
        return False
    
    logger.info(f"RTSP URL retrieved: {rtsp_url[:50]}...")
    
    # Khởi tạo video stream với low-latency mode
    stream = VideoStream(rtsp_url, buffer_size=1, frame_callback=None, 
                        low_latency=low_latency, drop_old_frames=drop_old_frames)
    
    if not stream.connect():
        logger.error("Failed to connect to RTSP stream")
        return False
    
    stream_info = stream.get_stream_info()
    actual_fps = stream_info.get('fps', 30.0)
    if actual_fps <= 0:
        actual_fps = 30.0  # Fallback
    
    # Tính toán frame skip tối ưu nếu auto_optimize được bật
    optimal_frame_skip = frame_skip
    if auto_optimize and frame_skip == config.FRAME_SKIP:  # Chỉ auto nếu dùng giá trị mặc định
        # Công thức: Frame Skip = (Camera FPS / 10) - 1
        # Mục tiêu: Xử lý ~10 FPS (đủ cho real-time)
        calculated_skip = max(0, int((actual_fps / 10) - 1))
        if calculated_skip != frame_skip:
            logger.info(f"Auto-optimization: Camera FPS {actual_fps:.2f} → Optimal frame skip: {calculated_skip} (was {frame_skip})")
            logger.info(f"  This will process ~{actual_fps/(calculated_skip+1):.1f} FPS for face recognition")
            optimal_frame_skip = calculated_skip
    else:
        optimal_frame_skip = frame_skip
    
    processing_fps = actual_fps / (optimal_frame_skip + 1)
    logger.info(f"Stream connected: {stream_info.get('width')}x{stream_info.get('height')} @ {actual_fps:.2f} FPS")
    logger.info(f"Frame skip: {optimal_frame_skip} → Processing ~{processing_fps:.1f} FPS for face recognition")
    logger.info(f"Starting attendance recognition. Press '{exit_key}' to exit.")
    logger.info("")
    
    # Lưu thời gian bắt đầu stream để tính timestamp chính xác
    stream_start_time = datetime.now()
    
    # Frame counter cho frame skip và tính timestamp
    frame_counter = 0
    processed_frame_counter = 0  # Chỉ đếm frame đã xử lý (không skip)
    
    # Thống kê để debug
    stats_period_start = datetime.now()
    stats_frames_processed = 0
    stats_faces_detected = 0
    stats_faces_recognized = 0
    stats_unknown_faces = 0
    
    try:
        if not stream.start():
            logger.error("Failed to start stream")
            return False
        
        logger.info("=" * 60)
        logger.info("STREAMING STARTED - Waiting for faces...")
        logger.info("=" * 60)
        logger.info("TIP: If no faces are detected, check:")
        logger.info("  1. MIN_FACE_SIZE in config.py (currently: {})".format(config.MIN_FACE_SIZE))
        logger.info("  2. Face detection model: {}".format(config.FACE_DETECTION_MODEL))
        logger.info("  3. Camera angle and lighting conditions")
        logger.info("  4. ROI settings (if enabled)")
        logger.info("=" * 60)
        
        while stream.is_running:
            ret, frame = stream.read_frame()
            
            if not ret or frame is None:
                logger.warning("Failed to read frame, retrying...")
                continue
            
            # Frame skip để giảm tải xử lý
            # Sử dụng optimal_frame_skip thay vì frame_skip
            # QUAN TRỌNG: Luôn xử lý frame để phát hiện người và hiển thị bounding box
            should_process = (optimal_frame_skip == 0) or (frame_counter % (optimal_frame_skip + 1) == 0)
            
            if should_process:
                # Xử lý frame với face recognition
                # Sử dụng processed_frame_counter để tính timestamp chính xác
                display_frame = video_processor.process_frame(
                    frame, 
                    processed_frame_counter, 
                    fps=actual_fps,
                    start_time=stream_start_time
                )
                processed_frame_counter += 1
                stats_frames_processed += 1
                
                # Cập nhật thống kê từ video_processor
                current_stats = video_processor.get_stats()
                stats_faces_detected = current_stats.get('total_checked_in', 0) + current_stats.get('total_unknown', 0) + current_stats.get('total_duplicate', 0)
                stats_faces_recognized = current_stats.get('total_checked_in', 0)
                stats_unknown_faces = current_stats.get('total_unknown', 0)
                
                # Log thống kê định kỳ (mỗi 5 giây)
                elapsed = (datetime.now() - stats_period_start).total_seconds()
                if elapsed >= 5.0:
                    logger.info(f"Statistics (last {elapsed:.1f}s): {stats_frames_processed} frames processed, "
                              f"{stats_faces_detected} faces detected ({stats_faces_recognized} recognized, {stats_unknown_faces} unknown)")
                    stats_period_start = datetime.now()
                    stats_frames_processed = 0
                
                frame_counter += 1
            else:
                # Frame không được xử lý - vẫn cần vẽ panel cố định
                # NHƯNG: Có thể vẫn cần hiển thị bounding box từ frame trước đó
                # Tạm thời: Vẽ panel trên frame gốc
                frame_counter += 1
                display_frame = frame.copy()
                
                # Vẽ panel cố định trên frame không được xử lý
                if config.ENABLE_OVERLAY:
                    from visualization import draw_statistics
                    display_frame = draw_statistics(display_frame, video_processor.get_stats())
            
            # Hiển thị frame
            cv2.imshow(window_name, display_frame)
            
            # Check for exit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord(exit_key):
                logger.info(f"Exit key '{exit_key}' pressed")
                break
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during streaming: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        stream.stop()
        cv2.destroyAllWindows()
        logger.info("Stream ended")
    
    return True


def initialize_components(args) -> tuple:
    """
    Initialize all components (EmployeeManager, FaceRecognizer, VideoProcessor).
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Tuple of (employee_manager, face_recognizer, video_processor, registered_employees)
    """
    from employee_manager import EmployeeManager
    from face_recognizer import FaceRecognizer
    from video_processor import VideoProcessor
    
    logger = logging.getLogger(__name__)
    
    # Step 1: Load danh sach nhan vien
    print("Buoc 1: Load danh sach nhan vien...")
    employee_manager = EmployeeManager(args.employees)
    employee_count = employee_manager.load_employees()
    
    if employee_count == 0:
        logger.error("Khong co nhan vien nao duoc load. Thoat chuong trinh.")
        sys.exit(1)
    
    print()
    
    # Step 2: Load danh sach dang ky
    print("Buoc 2: Load danh sach dang ky...")
    registered_employees = load_registered_employees(args.registered)
    print()
    
    # Step 3: Khoi tao FaceRecognizer
    print("Buoc 3: Khoi tao Face Recognizer...")
    face_recognizer = FaceRecognizer(employee_manager)
    print("OK: Face Recognizer da san sang")
    
    # Log thông tin cấu hình
    logger.info(f"Face recognition configuration:")
    logger.info(f"  - Tolerance (threshold): {config.FACE_RECOGNITION_TOLERANCE}")
    logger.info(f"  - Detection model: {config.FACE_DETECTION_MODEL}")
    logger.info(f"  - Detection upsample: {config.FACE_DETECTION_UPSAMPLE}")
    logger.info(f"  - Min face size: {config.MIN_FACE_SIZE}")
    logger.info(f"  - Employees loaded: {employee_manager.get_count()}")
    print()
    
    # Step 4: Khoi tao VideoProcessor
    print("Buoc 4: Khoi tao Video Processor...")
    video_processor = VideoProcessor(face_recognizer, registered_employees)
    print("OK: Video Processor da san sang")
    print()
    
    return employee_manager, face_recognizer, video_processor, registered_employees


def connect_to_camera(args) -> CameraController:
    """
    Connect to camera and return controller.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Connected CameraController instance
    """
    logger = logging.getLogger(__name__)
    
    print("Buoc 5: Ket noi camera...")
    controller = CameraController(
        host=args.host,
        port=args.port,
        username=args.username,
        password=args.password
    )
    
    if not controller.connect():
        logger.error("Failed to connect to camera. Please check:")
        logger.error("  - Camera IP address and port are correct")
        logger.error("  - Username and password are correct")
        logger.error("  - Camera supports ONVIF protocol")
        logger.error("  - Network/VPN connection is active")
        sys.exit(1)
    
    print("OK: Da ket noi camera")
    print()
    
    return controller


def save_results(video_processor: VideoProcessor, registered_employees: list,
                 employee_manager, args, start_time: datetime, end_time: datetime):
    """
    Analyze and save results to files.
    
    Args:
        video_processor: VideoProcessor instance
        registered_employees: List of registered employee IDs
        employee_manager: EmployeeManager instance
        args: Parsed command line arguments
        start_time: Session start time
        end_time: Session end time
    """
    from statistics import Statistics
    
    logger = logging.getLogger(__name__)
    
    print()
    print("Buoc 7: Phan tich va luu ket qua...")
    logger.info("Step 7: Analyzing and saving results...")
    
    # Tính thời gian chạy
    duration = end_time - start_time
    duration_seconds = duration.total_seconds()
    hours = int(duration_seconds // 3600)
    minutes = int((duration_seconds % 3600) // 60)
    seconds = int(duration_seconds % 60)
    milliseconds = int((duration_seconds % 1) * 1000)
    
    # Format thời gian
    if hours > 0:
        duration_str = f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        duration_str = f"{minutes}m {seconds}s"
    else:
        duration_str = f"{seconds}.{milliseconds:03d}s"
    
    video_stats = video_processor.get_stats()
    statistics = Statistics(registered_employees)
    analysis = statistics.analyze(video_stats, employee_manager)
    
    # Thêm thông tin thời gian vào analysis
    analysis['session_info'] = {
        'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
        'duration_seconds': duration_seconds,
        'duration_formatted': duration_str
    }
    
    # Tạo tên file output
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    camera_name = f"camera_{args.host.replace('.', '_')}"
    
    # Lưu báo cáo text
    report_file = os.path.join(args.output, "reports", f"report_{camera_name}_{date_str}.txt")
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    statistics.save_report(analysis, report_file, f"Camera {args.host}", 
                          start_time=start_time, end_time=end_time, duration_str=duration_str)
    logger.info(f"Report saved: {report_file}")
    
    # Lưu JSON
    json_file = os.path.join(args.output, "results", f"result_{camera_name}_{date_str}.json")
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    statistics.save_json(analysis, json_file)
    logger.info(f"JSON saved: {json_file}")
    
    # Lưu CSV
    csv_file = os.path.join(args.output, "results", f"result_{camera_name}_{date_str}.csv")
    statistics.save_csv(analysis, csv_file)
    logger.info(f"CSV saved: {csv_file}")
    
    # Log kết quả tóm tắt
    logger.info("=" * 60)
    logger.info("SESSION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Duration: {duration_str}")
    logger.info(f"Total people ate: {analysis['total_people_ate']}")
    logger.info(f"Registered and ate: {analysis['summary']['registered_and_ate_count']}")
    logger.info(f"Not registered but ate: {analysis['summary']['not_registered_but_ate_count']}")
    logger.info(f"Registered but not ate: {analysis['summary']['registered_but_not_ate_count']}")
    logger.info(f"Total checked in: {video_stats.get('total_checked_in', 0)}")
    logger.info(f"Total unknown: {video_stats.get('total_unknown', 0)}")
    logger.info(f"Total duplicate: {video_stats.get('total_duplicate', 0)}")
    logger.info("=" * 60)
    
    print()
    
    # Hiển thị báo cáo tóm tắt
    print("=" * 60)
    print(statistics.generate_report(analysis, f"Camera {args.host}", 
                                     start_time=start_time, end_time=end_time, duration_str=duration_str))
    print()
    
    print("OK: Hoan thanh!")
    print(f"Thoi gian chay: {duration_str}")
    print(f"Ket qua da duoc luu tai: {args.output}")
    print(f"Log file: {config.LOG_FILE}")
    
    logger.info("=" * 60)
    logger.info("CAMERA ATTENDANCE SYSTEM - SESSION END")
    logger.info(f"Total duration: {duration_str}")
    logger.info("=" * 60)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Hệ thống điểm danh nhà ăn từ IP camera - Nhận diện khuôn mặt real-time',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sử dụng cấu hình từ config.py
  python camera_attendance.py --registered data/registered/registered_today.json
  
  # Override với tham số từ command line
  python camera_attendance.py --host 172.16.20.41 --username admin --password pass --registered data/registered/registered_today.json
        """
    )
    
    # Camera arguments
    parser.add_argument(
        '--host',
        type=str,
        default=config.CAMERA_HOST,
        help=f'Camera IP address (default: {config.CAMERA_HOST})'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=config.CAMERA_PORT,
        help=f'ONVIF service port (default: {config.CAMERA_PORT})'
    )
    
    parser.add_argument(
        '--username',
        type=str,
        default=config.CAMERA_USERNAME,
        help=f'Camera username (default: {config.CAMERA_USERNAME})'
    )
    
    parser.add_argument(
        '--password',
        type=str,
        default=config.CAMERA_PASSWORD,
        help='Camera password (default: from config.py)'
    )
    
    parser.add_argument(
        '--profile-token',
        type=str,
        default=None,
        help='Specific profile token to use (default: first available)'
    )
    
    parser.add_argument(
        '--stream-only',
        action='store_true',
        help='Chỉ hiển thị video stream (không nhận diện khuôn mặt) - giao diện giống test'
    )
    
    # Attendance arguments
    parser.add_argument(
        '--registered',
        type=str,
        required=False,
        help='Đường dẫn đến file JSON danh sách đăng ký (bắt buộc nếu không dùng --stream-only)'
    )
    
    parser.add_argument(
        '--employees',
        type=str,
        default=getattr(config, 'EMPLOYEES_DIR', os.path.join('data', 'employees')),
        help=f'Đường dẫn đến thư mục ảnh nhân viên (mặc định: {getattr(config, "EMPLOYEES_DIR", os.path.join("data", "employees"))})'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=getattr(config, 'OUTPUT_DIR', 'output'),
        help=f'Thư mục lưu kết quả (mặc định: {getattr(config, "OUTPUT_DIR", "output")})'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=getattr(config, 'FACE_RECOGNITION_TOLERANCE', 0.5),
        help=f'Ngưỡng nhận diện (mặc định: {getattr(config, "FACE_RECOGNITION_TOLERANCE", 0.5)})'
    )
    
    parser.add_argument(
        '--frame-skip',
        type=int,
        default=getattr(config, 'FRAME_SKIP', 3),
        help=f'Số frame bỏ qua giữa 2 lần xử lý (mặc định: {getattr(config, "FRAME_SKIP", 3)})'
    )
    
    parser.add_argument(
        '--no-overlay',
        action='store_true',
        help='Tắt overlay (ngay cả khi đang stream)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--no-low-latency',
        action='store_true',
        help='Tắt chế độ low-latency (có thể tăng delay nhưng ổn định hơn)'
    )
    
    parser.add_argument(
        '--no-drop-frames',
        action='store_true',
        help='Tắt frame dropping (xử lý tất cả frame, có thể tăng delay)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file = getattr(config, 'LOG_FILE', os.path.join(getattr(config, 'LOGS_DIR', 'logs'), 'kitchen_cam.log'))
    setup_logging(log_level, log_file)
    logger = logging.getLogger(__name__)
    
    # Validate camera connection (luôn cần)
    try:
        if not args.host or len(args.host.strip()) == 0:
            raise ValueError("Camera host cannot be empty")
        
        if args.port < 1 or args.port > 65535:
            raise ValueError(f"Port must be between 1 and 65535, got: {args.port}")
    except ValueError as e:
        logger.error(f"Camera validation failed: {str(e)}")
        sys.exit(1)
    
    # Nếu chế độ stream-only, chỉ cần kết nối camera và hiển thị
    if args.stream_only:
        print("=" * 60)
        print("  TEST CAMERA STREAM - ONVIF/RTSP")
        print("=" * 60)
        print()
        
        # Kết nối camera
        controller = CameraController(
            host=args.host,
            port=args.port,
            username=args.username,
            password=args.password
        )
        
        logger.info(f"Đang kết nối đến camera {args.host}:{args.port}...")
        if not controller.connect():
            logger.error("Không thể kết nối đến camera!")
            logger.error("Vui lòng kiểm tra:")
            logger.error("  - Địa chỉ IP camera")
            logger.error("  - Port ONVIF (thường là 80, 8080, hoặc 554)")
            logger.error("  - Username và password")
            logger.error("  - Camera có hỗ trợ ONVIF không")
            logger.error("  - Kết nối mạng/VPN")
            sys.exit(1)
        
        logger.info("✓ Đã kết nối camera thành công!")
        print()
        
        # Hiển thị stream
        stream_simple(controller, window_name="Camera Stream", exit_key='q')
        
        # Ngắt kết nối
        controller.disconnect()
        logger.info("Đã ngắt kết nối camera")
        
        print()
        logger.info("=" * 60)
        logger.info("TEST HOÀN TẤT")
        logger.info("=" * 60)
        return
    
    # Chế độ nhận diện khuôn mặt - cần validate đầy đủ
    if not args.registered:
        logger.error("--registered là bắt buộc khi không dùng --stream-only")
        sys.exit(1)
    
    # Input validation cho chế độ nhận diện
    try:
        # Validate registered file
        validate_file_path(args.registered, "file")
        validate_json_file(args.registered)
        
        # Validate employees directory
        validate_file_path(args.employees, "directory")
        
        # Validate output directory (create if not exists)
        if not os.path.exists(args.output):
            os.makedirs(args.output, exist_ok=True)
        elif not os.path.isdir(args.output):
            raise ValueError(f"Output path is not a directory: {args.output}")
        
        # Validate threshold range
        if args.threshold < 0 or args.threshold > 1:
            raise ValueError(f"Threshold must be between 0 and 1, got: {args.threshold}")
        
        # Validate frame skip
        if args.frame_skip < 0:
            raise ValueError(f"Frame skip must be >= 0, got: {args.frame_skip}")
        
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Input validation failed: {str(e)}")
        sys.exit(1)
    
    # Log session start
    # Lưu thời gian bắt đầu
    start_time = datetime.now()
    
    logger.info("=" * 60)
    logger.info("CAMERA ATTENDANCE SYSTEM - SESSION START")
    logger.info("=" * 60)
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Camera: {args.host}:{args.port}")
    logger.info(f"Registered file: {args.registered}")
    logger.info(f"Employees directory: {args.employees}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Frame skip: {args.frame_skip}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"ROI enabled: {config.ROI_ENABLED}")
    logger.info(f"Overlay enabled: {config.ENABLE_OVERLAY}")
    logger.info("=" * 60)
    
    # Cập nhật config nếu có tham số từ CLI
    if args.threshold != config.FACE_RECOGNITION_TOLERANCE:
        config.FACE_RECOGNITION_TOLERANCE = args.threshold
        logger.info(f"Updated threshold: {args.threshold}")
    
    if args.no_overlay:
        config.ENABLE_OVERLAY = False
        logger.info("Overlay disabled via --no-overlay flag")
    else:
        # Đảm bảo overlay được bật (mặc định)
        config.ENABLE_OVERLAY = True
        logger.info("Overlay enabled (default)")
    
    # Set UTF-8 encoding for Windows console
    import sys
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except:
            pass
    
    print("=" * 60)
    print("  HE THONG DIEM DANH NHA AN TU IP CAMERA")
    print("=" * 60)
    print()
    
    # Initialize components
    employee_manager, face_recognizer, video_processor, registered_employees = initialize_components(args)
    
    # Connect to camera
    controller = connect_to_camera(args)
    
    # Step 6: Bat dau stream va nhan dien
    print("Buoc 6: Bat dau stream va nhan dien...")
    print(f"  Press 'q' to quit")
    print()
    
    try:
        # Stream với face recognition
        # Xác định low-latency và drop frames từ config hoặc command line
        low_latency = config.LOW_LATENCY_MODE and not args.no_low_latency
        drop_frames = config.DROP_OLD_FRAMES and not args.no_drop_frames
        
        success = stream_with_attendance(
            controller,
            video_processor,
            window_name="Camera Attendance",
            exit_key='q',
            frame_skip=args.frame_skip,
            auto_optimize=True,  # Tự động tối ưu frame skip dựa trên FPS camera
            low_latency=low_latency,
            drop_old_frames=drop_frames
        )
        
        if not success:
            logger.error("Streaming failed")
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during streaming: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        controller.disconnect()
        logger.info("Disconnected from camera")
    
    # Save results
    end_time = datetime.now()
    save_results(video_processor, registered_employees, employee_manager, args, start_time, end_time)


if __name__ == "__main__":
    main()
