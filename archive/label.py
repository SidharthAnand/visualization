from analysis_general import sensor_count, separate_sensors, data_converter
from datetime import datetime
import pygame
import numpy as np
import argparse
import ast


def label_data(path, write_path, sensor_id=None, begin_frame=-1, begin_timestamp=0):
    """
    :param path: Path to raw sensor data, to be converted to the accepted format
    :param write_path: Where the output will be written
    :param sensor_id: list of sensor deviceName strings. If not provided they will be collected automatically
    :param begin_frame: Start the labeling from this line in the file
    :param begin_timestamp: Start the labeling from this timestamp

    :return: .json files formatted as below.

    SAMPLE OUTPUT:
        {"deviceName": "5",
         "timestamp": 1598290369000,
         "detectionsLocal": [[0.5625, 0.4375]],
         "detecionsWorld": [[1.8725, 3.0875]],
         "frame": 3}

    NOTE: Frames are not always synchronized. In fact, they are oftentimes not. Always rely on timestamp rather than
            frame number. The frame number is just the line in the text file that has just been analyzed.
            Thus, it does not contain much useful information.

    In the case that you get to the end of the sensor data file and try to continue, you will be automatically
    redirected to the next sensor data file in the sequence. If you try to continue after finishing the last file,
    the program will quit.
    """
    data_converter(path, path[:-4] + "_converted.txt")
    converted = path[:-4] + "_converted.txt"

    if not sensor_id:
        sensor_id = sensor_count(converted)[0]

    new_paths = separate_sensors(converted, [sensor_id])

    # Set up Pygame
    pygame.init()
    aspect = (800, 660)
    surf_size = (450, 450)
    black = (0, 0, 0)
    white = (255, 255, 255)
    red = (255, 0, 0)
    green = (0, 100, 40)
    d_red = (150, 0, 0)
    yellow = (150, 150, 0)
    start_x, start_y = (50, 50)
    disp = pygame.display.set_mode(aspect)
    font = pygame.font.Font('freesansbold.ttf', 14)
    pygame.display.set_caption('Butlr Sensor Data Manual Labeling Tool')

    disp.fill(black)
    # Initialize indices, buttons, and "done" flags before starting
    index = begin_frame
    file_end = True
    done = True
    running = True
    times = {}
    json_list = []

    while running:
        if file_end:
            file_end = False
            sensor = sensor_id
            index = begin_frame
            with open(new_paths[sensor], 'r') as f:
                text = f.readlines()

        # Reset everything and start over with a new grayscale frame
        if done:
            disp.fill(black)
            index += 1
            done = False
            try:
                output_dict = ast.literal_eval(text[index])
                timestamp = output_dict['timestamp']
                while timestamp < begin_timestamp:
                    index += 1
                    output_dict = ast.literal_eval(text[index])
                    timestamp = output_dict['timestamp']
            except IndexError:
                file_end = True
                done = True
                continue
            if timestamp not in times.keys():
                times[timestamp] = []

            # Format placement of text and buttons
            date = datetime.fromtimestamp((timestamp / 1000) + (3600 * 3))
            time_text = font.render("Time: " + str(date), True, white)
            time_textRect = time_text.get_rect()
            time_textRect.topleft = (20, 24)
            sensor_text = font.render("Sensor ID: " + str(sensor), True, white)
            sensor_rect = sensor_text.get_rect()
            sensor_rect.topleft = (20, 8)
            cont_text = font.render(">", True, black, None)
            cont_rect = cont_text.get_rect()
            cont_rect.center = (680, 380)
            clear_text = font.render("Clear", True, white, None)
            clear_rect = clear_text.get_rect()
            clear_rect.center = (635, 470)
            back_text = font.render("<", True, black, None)
            back_rect = back_text.get_rect()
            back_rect.center = (590, 380)
            rw_text = font.render("<<", True, black, None)
            rw_rect = rw_text.get_rect()
            rw_rect.center = (590, 345)
            ff_text = font.render(">>", True, black, None)
            ff_rect = ff_text.get_rect()
            ff_rect.center = (680, 345)

            write_text = font.render("Write File", True, black, None)
            write_rect = write_text.get_rect()
            write_rect.center = (635, 80)

            # Get data and create pygame surface
            data = output_dict['data']
            data = [min(max(x, 80.001), 147.999) for x in data]
            data = np.asarray(data).reshape((8, 8)) * 0.25
            surf = pygame.surfarray.make_surface(gray(data))
            surf = pygame.transform.scale(surf, surf_size)
            surf = pygame.transform.rotate(surf, 90)

            for cd in times[timestamp]:
                cd_scale = (cd[0] * surf_size[0], cd[1] * surf_size[1])
                pygame.draw.circle(surf, red, cd_scale, 10, width=1)

            # Show
            clear = pygame.draw.rect(disp, d_red, (560, 440, 150, 60))
            write = pygame.draw.rect(disp, (169, 169, 169), (560, 50, 150, 60))

            play = pygame.draw.rect(disp, (100, 100, 100), (560, 250, 150, 60))
            pause = pygame.draw.rect(disp, (100, 100, 100), (560, 170, 150, 60))
            skip_forward = pygame.draw.rect(disp, (100, 100, 100), (560, 330, 60, 30))
            skip_back = pygame.draw.rect(disp, (100, 100, 100), (650, 330, 60, 30))

            cont_button = pygame.draw.rect(disp, (100, 100, 100), (650, 365, 60, 30))
            back = pygame.draw.rect(disp, (100, 100, 100), (560, 365, 60, 30))

            disp.blit(surf, (start_x, start_y))
            disp.blit(time_text, time_textRect)
            disp.blit(sensor_text, sensor_rect)
            disp.blit(clear_text, clear_rect)
            disp.blit(cont_text, cont_rect)
            disp.blit(back_text, back_rect)
            disp.blit(write_text, write_rect)
            disp.blit(rw_text, rw_rect)
            disp.blit(ff_text, ff_rect)

            logo = pygame.image.load("logo/butlr.logo.png")
            disp.blit(logo, (20, 520))

            pygame.display.update()

        # Waiting for mouse clicks or quit
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = (pygame.mouse.get_pos()[0] - start_x, pygame.mouse.get_pos()[1] - start_y)
                # print(pos)
                if 0 <= pos[0] <= surf_size[0] and 0 <= pos[1] <= surf_size[1]:
                    out_coord = ((pos[0] / surf_size[0]), pos[1] / surf_size[1])
                    # print(out_coord)
                    times[timestamp].append(out_coord)
                    pygame.draw.circle(surf, red, pos, 10, width=1)

                    disp.blit(surf, (start_x, start_y))
                    pygame.display.update()
                else:
                    pos1 = pygame.mouse.get_pos()

                    # Continue on to the next frame
                    if cont_button.collidepoint(pos1):
                        return_dict = {"deviceName": sensor,
                                       "timestamp": timestamp,
                                       "utcSecs": output_dict['utcSecs'],
                                       "utcUsecs": output_dict['utcUsecs'],
                                       "detectionsLocal": times[timestamp],
                                       "detectionsWorld": [],
                                       "frame": index}
                        # print(return_dict)
                        json_list.append(return_dict)
                        done = True

                    # Clear the frame and start over
                    elif clear.collidepoint(pos1):
                        surf = pygame.surfarray.make_surface(gray(data))
                        surf = pygame.transform.scale(surf, surf_size)
                        surf = pygame.transform.rotate(surf, 90)
                        disp.blit(surf, (start_x, start_y))
                        times[timestamp] = []
                        pygame.display.update()

                    # Go back one frame
                    elif back.collidepoint(pos1):
                        try:
                            json_list.pop(-1)
                        except IndexError:
                            pass
                        index -= 2
                        done = True

                    # Write the file
                    elif write.collidepoint(pos1):
                        print("=============== OUTPUT ================")
                        with open(write_path, 'a+') as out_file:
                            for j in json_list:
                                print(j)
                                out_file.write(str(j))
                                out_file.write('\n')
                        file_end = True
                        done = True
                        running = False

                    # TODO: Talk to Arihan about how to use the API to stream the data in text format.
                    # Begin streaming the frames in sequence
                    elif play.collidepoint(pos1):
                        raise NotImplementedError

                    # Stop streaming the frames
                    elif pause.collidepoint(pos1):
                        raise NotImplementedError

                    elif skip_back.collidepoint(pos1):
                        try:
                            for i in range(1, 11):
                                json_list.pop(-i)
                        except IndexError:
                            pass
                        index -= 11
                        done = True
                    elif skip_forward.collidepoint(pos1):
                        pass

            if event.type == pygame.QUIT:
                # print(json_list)
                running = False


def gray(im):
    lo = 15
    hi = 46
    lo_pot = np.percentile(im, 2) - 3
    hi_pot = np.percentile(im, 99) + 7
    if lo_pot < lo:
        lo = lo_pot
    if hi_pot > hi:
        hi = hi_pot
    # print(lo, hi)
    im = ((255 / (hi - lo)) * im) - lo
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', default="Data/seamless/data_seamless_input_gmt_2020-08-13_15.txt")
    parser.add_argument('-id', default='00-17-0d-00-00-59-4d-21')
    parser.add_argument('-idx', default=-1)
    parser.add_argument('-stmp', default=0)
    parser.add_argument('-p_out')
    args = parser.parse_args()
    path = args.path
    ids = args.id
    frame = args.idx
    stamp = int(args.stmp)

    out_path = args.p_out

    label_data(path, out_path, sensor_id=ids, begin_frame=frame, begin_timestamp=stamp)


if __name__ == "__main__":
    main()
