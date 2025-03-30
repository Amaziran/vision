import cv2
import numpy as np
from collections import deque
import serial
from multiprocessing import Process, Queue, Manager
from time import sleep

# 串口配置
ser = serial.Serial("/dev/ttyS3", 9600)

if not ser.isOpen():
    print("open failed")
else:
    print("open success:")
    print(ser)

class CircleDetector:
    def __init__(self):
        self.color_ranges = {
            "red": ((132, 48, 70), (179, 212, 255)),
            "green": ((34, 39, 174), (53, 105, 223)),
            "blue": ((114, 66, 127), (130, 162, 223)),
        }
        # 存储每个颜色的圆心坐标队列，最大长度为 12
        self.circle_centers_queue = {
            "red": deque(maxlen=12),
            "green": deque(maxlen=12),
            "blue": deque(maxlen=12)
        }
        self.color_to_code = {
            "red": '2',
            "green": '1',
            "blue": '0'
        }

    def detect_circles(self, frame, target_color_code=None):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detected_frame = frame.copy()
        frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)

        for color, (lower, upper) in self.color_ranges.items():
            color_code = self.color_to_code[color]
            if target_color_code is not None and target_color_code != color_code:
                continue

            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            # 生成颜色掩模
            mask = cv2.inRange(hsv, lower, upper)
            #cv2.imshow(f'{color} Mask', mask)

            # 对掩模进行高斯模糊
            blurred_mask = cv2.GaussianBlur(mask, (9, 9), 2)
            #cv2.imshow(f'{color} Blurred Mask', blurred_mask)

            # 霍夫梯度圆检测
            circles = cv2.HoughCircles(
                blurred_mask,
                cv2.HOUGH_GRADIENT,
                dp=1.1,
                minDist=50,
                param1=30,
                param2=10,
                minRadius=66,
                maxRadius=70
            )

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                # 只取第一个检测到的圆
                x, y, r = circles[0]
                self.circle_centers_queue[color].append((x, y))

                # 根据颜色设置绘制颜色
                if color == "red":
                    draw_color = (0, 0, 255)
                elif color == "green":
                    draw_color = (0, 255, 0)
                elif color == "blue":
                    draw_color = (255, 0, 0)

                # 如果队列至少有 6 个元素，计算加权平均圆心
                if len(self.circle_centers_queue[color]) >= 6:
                    # 取最近的 6 个元素
                    recent_centers = list(self.circle_centers_queue[color])[-6:]
                    weights = np.linspace(0.1, 1, 6)
                    weights = weights / weights.sum()
                    weighted_sum_x = 0
                    weighted_sum_y = 0
                    for i, (cx, cy) in enumerate(recent_centers):
                        weighted_sum_x += cx * weights[i]
                        weighted_sum_y += cy * weights[i]
                    weighted_center_x = int(weighted_sum_x)
                    weighted_center_y = int(weighted_sum_y)

                    # 绘制加权平均后的圆心
                    cv2.circle(detected_frame, (weighted_center_x, weighted_center_y), r, draw_color, 2)
                    # 输出到终端
                    print(f"{color} 颜色的加权平均圆心坐标: ({weighted_center_x}, {weighted_center_y})")

                    # 计算相对坐标
                    relative_x = weighted_center_x - frame_center[0]
                    relative_y = frame_center[1] - weighted_center_y

                    res = f"X {color_code} {((relative_x // 10 * 10) // 10)} {-((relative_y // 10 * 10) // 10)+6} Y"
                    print(f"Generated data: {res}")

                    try:
                        # 通过串口发送数据
                        ser.write(res.encode("utf-8"))
                        print(f"Sent data to serial port: {res}")
                    except Exception as e:
                        print(f"Error sending data to serial port: {e}")

        return detected_frame

def uart_manager(rx_queue, tx_queue):
    """
    专门管理串口通信的进程
    :param rx_queue: 用于接收数据的队列
    :param tx_queue: 用于发送数据的队列
    """
    while True:
        # 检查是否有数据需要发送
        if not tx_queue.empty():
            data_to_send = tx_queue.get()
            if isinstance(data_to_send, bytes):
                ser.write(data_to_send)
                print(f"Sent: {data_to_send}")
            else:
                ser.write(data_to_send.encode("utf-8"))
                print(f"Sent: {data_to_send}")

        # 检查是否有数据可读取
        count = ser.inWaiting()
        if count > 0:
            recv = ser.read(count)
            rx_queue.put(recv)
            print(f"Received: {recv.decode('utf-8')}")

        sleep(0.01)

def data_processor(rx_queue, tx_queue):
    detector = CircleDetector()
    cap = cv2.VideoCapture(9)
    ret, frame = cap.read()  # 在循环外部读取第一帧

    while True:
        if not rx_queue.empty():
            data = rx_queue.get()
            realData = data.decode('utf-8')
            print(f"Processing received data: {realData}")

            if ret:  # 检查是否成功读取帧
                if "0" in realData:
                    detector.detect_circles(frame, target_color_code='0')
                elif "1" in realData:
                    detector.detect_circles(frame, target_color_code='1')
                elif "2" in realData:
                    detector.detect_circles(frame, target_color_code='2')
                else:
                    res = {"type": "error", "params": {"MESSAGE": "YOUR TYPE NOT HAVE."}}
                    tx_queue.put(str(res))

                # 读取下一帧
                ret, frame = cap.read()
                if not ret:
                    print("无法读取帧，退出...")
                    break

        # 显示原始帧
        #if ret:
            #cv2.imshow('Original Frame', frame)
            # 按 'q' 键退出循环
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 队列初始化
    rx_queue = Queue()
    tx_queue = Queue()

    # 创建和启动进程
    uart_process = Process(target=uart_manager, args=(rx_queue, tx_queue))
    processor_process = Process(target=data_processor, args=(rx_queue, tx_queue))

    uart_process.start()
    processor_process.start()

    uart_process.join()
    processor_process.join()

    ser.close()