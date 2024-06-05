from PIL import Image, ImageDraw, ImageFont

class PolygonVisualizer:
    def __init__(self, canvas:Image.Image=None):
        self.canvas = canvas
        self.vis_final = None

    def canvas_from_patches(self, image_batch:list, offset_xs:list, offset_ys:list):
        h = max([image.size[0] + offset_x for image, offset_x in zip(image_batch, offset_xs)])
        w = max([image.size[1] + offset_y for image, offset_y in zip(image_batch, offset_ys)])
        self.canvas = Image.new("RGB", (h,w))
        for image, offset_x, offset_y in zip(image_batch, offset_xs, offset_ys):
            self.canvas.paste(image, (offset_x, offset_y))

    def canvas_from_image(self, image:Image.Image):
        self.canvas = image.copy()

    def _draw_polygon(self, polygon, color, alpha=0.5):
        draw = ImageDraw.Draw(self.vis_final, "RGBA")
        draw.polygon(polygon, tuple([int(255*c) for c in color] + [int(255*alpha)]), outline=tuple([int(255*c) for c in color]))

    def _draw_text(self, text, pos, color, horiz_align="left", font_size=2):
        draw = ImageDraw.Draw(self.vis_final)
        font = ImageFont.truetype("arial.ttf", font_size)
        draw.text([pos[0], pos[1]-10], text, tuple([int(255*c) for c in color]), align=horiz_align, font=font)

    def _draw_polyline(self, poly_x, poly_y, color):
        draw = ImageDraw.Draw(self.vis_final)
        draw.line(list(zip(poly_x, poly_y)), fill=tuple([int(255*c) for c in color]), width=2)

    def _draw_bezier(self, bezier_x, bezier_y, color):
        from . import bezier_utils as bezier_utils
        poly_x, poly_y = bezier_utils.bezier_to_polyline(bezier_x, bezier_y)
        self._draw_polyline(poly_x, poly_y, color)

        # Draw the first point
        draw = ImageDraw.Draw(self.vis_final)
        draw.ellipse([bezier_x[0]-2, bezier_y[0]-2, bezier_x[0]+2, bezier_y[0]+2], fill=tuple([int(255*c) for c in color]))

    def draw(self, json_list:list, colors = None, alpha = 0.5, canvas = None, text_override = None) -> Image.Image:
        if self.canvas == None:
            print("No canvas loaded.")
            return
        if canvas != None:
            self.canvas = canvas
        self.vis_final = self.canvas.copy()

        if colors is None:
            colors = [(0.1, 0.2, 0.5)] * len(json_list)

        alpha = alpha
        for json, color in zip(json_list, colors):
            polygon_x = json["polygon_x"]
            polygon_y = json["polygon_y"]
            polygon = [(x, y) for x, y in zip(polygon_x, polygon_y)]

            self._draw_polygon(polygon, color, alpha=alpha)

            # draw text in the top left corner
            word = json["text"]
            score = json["score"]
            text = "{:.3f}: {}".format(score, word)
            lighter_color = [c + 0.5*(1-c) for c in color]
            black = (0, 0, 0)
            text_pos = polygon[0]
            horiz_align = "left"
            font_size = 20

            if text_override == None:
                self._draw_text(text, text_pos, black, horiz_align=horiz_align, font_size=font_size)

            if 'upper_bezier_pts' in json:
                ub_color = (0.5, 0.1, 0.2)
                u_bezier_pts = json['upper_bezier_pts']
                u_bezier_pts_x = [pt[0] for pt in u_bezier_pts]
                u_bezier_pts_y = [pt[1] for pt in u_bezier_pts]
                self._draw_bezier(u_bezier_pts_x, u_bezier_pts_y, ub_color)

            if 'lower_bezier_pts' in json:
                lb_color = (0.5, 0.2, 0.1)
                l_bezier_pts = json['lower_bezier_pts']
                l_bezier_pts_x = [pt[0] for pt in l_bezier_pts]
                l_bezier_pts_y = [pt[1] for pt in l_bezier_pts]
                self._draw_bezier(l_bezier_pts_x, l_bezier_pts_y, lb_color)

            if 'center_bezier_pts' in json:
                cb_color = (0.1, 0.5, 0.2)
                c_bezier_pts = json['center_bezier_pts']
                c_bezier_pts_x = [pt[0] for pt in c_bezier_pts]
                c_bezier_pts_y = [pt[1] for pt in c_bezier_pts]
                self._draw_bezier(c_bezier_pts_x, c_bezier_pts_y, cb_color)

        if text_override != None:
            for text_draw in text_override:
                center = text_draw["center"]
                text = text_draw["text"]
                color = (0, 0, 0)
                self._draw_text(text, center, color, horiz_align="center", font_size=20)

        return self.vis_final
    
    def draw_multiple(self, json_list:list, text_override = None, colors_override = None) -> Image.Image:
        import numpy as np

        flattened_list = []
        colors = []
        if colors_override:
            for json, color in zip(json_list, colors_override):
                for i, detection in enumerate(json):
                    detection['text'] += f'_{i}'
                flattened_list.extend(json)
                colors.extend([color]*len(json))
        else:
            for json in json_list:
                color = np.random.rand(3)
                colors.extend([color]*len(json))
                for i, detection in enumerate(json):
                    detection['text'] += f'_{i}'
                flattened_list.extend(json)

        return self.draw(flattened_list, colors, text_override=text_override)
    
    def draw_toponyms(self, toponyms):
        import numpy as np

        flattened_list = []
        text_override = []
        colors = []
        for toponym in toponyms:
            text = toponym['text']
            center = toponym['center']
            group = toponym['group']
            color = np.random.rand(3)
            colors.extend([color]*len(group))

            flattened_list.extend(group)
            text_override.append({'text': text, 'center': center})
        
        return self.draw(flattened_list, colors=colors, text_override=toponyms)

    def save(self, output_path):
        self.vis_final.save(output_path)
        return
    
    def save_json(self, json_list:list, output_path):
        import pandas as pd
        df = pd.DataFrame(json_list)
        df.to_json(output_path)
        return