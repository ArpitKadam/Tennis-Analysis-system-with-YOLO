def convert_pixel_distance_to_meters(pixel_distance, reference_height_in_meters, reference_height_in_pixels):
    return (pixel_distance * reference_height_in_meters) / reference_height_in_pixels

def convert_meters_distance_to_pixels(meters, reference_height_in_meters, reference_height_in_pixels):
    return (meters * reference_height_in_pixels) / reference_height_in_meters
