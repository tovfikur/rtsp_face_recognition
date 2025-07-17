import subprocess
import sys
import os

def convert_rtsp_to_hls(rtsp_url, output_dir="hls_output", hls_time=2, hls_list_size=3):
    """
    Convert an RTSP stream to HLS format using FFmpeg.
    
    Args:
        rtsp_url (str): The RTSP stream URL (e.g., rtsp://your-camera-ip/stream)
        output_dir (str): Directory to store HLS output files
        hls_time (int): Duration of each HLS segment in seconds
        hls_list_size (int): Number of segments to keep in the playlist
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Construct FFmpeg command
        output_m3u8 = os.path.join(output_dir, "output.m3u8")
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", rtsp_url,
            "-c", "copy",
            "-f", "hls",
            "-hls_time", str(hls_time),
            "-hls_list_size", str(hls_list_size),
            "-hls_flags", "delete_segments",
            output_m3u8
        ]
        
        # Run FFmpeg command
        print(f"Starting conversion of {rtsp_url} to HLS...")
        process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        
        # Stream FFmpeg output in real-time
        for line in process.stderr:
            print(line.strip())
            
        process.wait()
        
        if process.returncode == 0:
            print(f"Conversion successful! HLS files are saved in {output_dir}")
        else:
            print(f"Conversion failed with return code {process.returncode}")
            
    except FileNotFoundError:
        print("Error: FFmpeg is not installed or not found in system PATH.")
        print("Please install FFmpeg and ensure it's accessible from the command line.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_rtsp_to_hls.py <rtsp_url> [output_dir]")
        print("Example: python convert_rtsp_to_hls.py rtsp://192.168.1.100:554/stream hls_output")
        sys.exit(1)
    
    rtsp_url = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "hls_output"
    
    convert_rtsp_to_hls(rtsp_url, output_dir)