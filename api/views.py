from django.shortcuts import render
from rest_framework.decorators import api_view
from django.http import HttpResponse
from django.http import JsonResponse
from django.views import View
from django.http import HttpResponse, HttpResponseNotFound
import base64
import numpy as np
import cv2
import re
import os


def decode_base64(data, altchars=b'+/'):
    """Decode base64, padding being optional.

    :param data: Base64 data as an ASCII byte string
    :returns: The decoded byte string.

    """
    data = re.sub(rb'[^a-zA-Z0-9%s]+' % altchars, b'', data)  # normalize
    missing_padding = len(data) % 4
    if missing_padding:
        data += b'=' * (4 - missing_padding)
    return base64.b64decode(data, altchars)

# Create your views here.


def main(request):
    return HttpResponse("Hello!")


@api_view(['GET', 'POST'])
def grayscale(request):
    # decode base64
    img_bs64 = decode_base64(request.data['image'].encode("utf-8"))
    im_arr = np.fromstring(img_bs64, dtype=np.uint8)
    img = cv2.imdecode(np.array(im_arr), cv2.IMREAD_UNCHANGED)

    # process image from base64
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # encode base64
    _, img_arr_decode = cv2.imencode('.jpg', gray)
    im_bytes_decode = img_arr_decode.tobytes()
    im_b64_out = base64.b64encode(im_bytes_decode).decode("utf-8")

    return JsonResponse({"data": im_b64_out})


@api_view(['GET', 'POST'])
def high_pass(request):
    # decode base64
    img_bs64 = decode_base64(request.data['image'].encode("utf-8"))
    im_arr = np.fromstring(img_bs64, dtype=np.uint8)
    img = cv2.imdecode(np.array(im_arr), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if(request.data['type'] == "Kernel 1"):
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    elif(request.data['type'] == "Kernel 2"):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    elif(request.data['type'] == "Kernel 3"):
        kernel = np.array([[0, -1, 0], [-1, 8, -1], [0, -1, 0]])
    elif(request.data['type'] == "Kernel 4"):
        kernel = np.array([[1, -2, 1], [-2, 5, -2], [1, -2, 1]])
    elif(request.data['type'] == "Kernel 5"):
        kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
    elif(request.data['type'] == "Kernel 6"):
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    elif(request.data['type'] == "Laplace"):
        kernel = (1.0 / 16) * np.array(
            [
                [0, 0, -1, 0, 0],
                [0, -1, -2, -1, 0],
                [-1, -2, 16, -2, -1],
                [0, -1, -2, -1, 0],
                [0, 0, -1, 0, 0],
            ]
        )
    elif(request.data['type'] == "Custom"):
        kernel = np.array(request.data['kernel'], dtype=np.int8)

    img_out = cv2.filter2D(img, -1, kernel)
    # encode base64
    _, img_arr_decode = cv2.imencode('.jpg', img_out)
    im_bytes_decode = img_arr_decode.tobytes()
    im_b64_out = base64.b64encode(im_bytes_decode).decode("utf-8")
    return JsonResponse({"data": im_b64_out})


@api_view(['GET', 'POST'])
def motion_blur(request):
    # decode base64
    img_bs64 = decode_base64(request.data['image'].encode("utf-8"))
    im_arr = np.fromstring(img_bs64, dtype=np.uint8)
    img = cv2.imdecode(np.array(im_arr), cv2.IMREAD_UNCHANGED)

    # motion blur
    kernel_size = 30
    kernel_v = np.zeros((kernel_size, kernel_size))
    kernel_h = np.copy(kernel_v)

    kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
    kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)

    kernel_v /= kernel_size
    kernel_h /= kernel_size
    vertical_mb = cv2.filter2D(img, -1, kernel_v)
    horizonal_mb = cv2.filter2D(img, -1, kernel_h)

    # encode base64
    if(request.data['type'] == "vertical"):
        _, img_arr_decode = cv2.imencode('.jpg', vertical_mb)
    else:
        _, img_arr_decode = cv2.imencode('.jpg', horizonal_mb)
    im_bytes_decode = img_arr_decode.tobytes()
    im_b64_out = base64.b64encode(im_bytes_decode).decode("utf-8")

    return JsonResponse({"data": im_b64_out})
