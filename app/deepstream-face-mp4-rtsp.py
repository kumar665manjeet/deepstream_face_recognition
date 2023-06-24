#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2020-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import sys
sys.path.append('../')
import configparser

import numpy as np

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GObject, Gst, GstRtspServer
from gi.repository import GObject, Gst
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.FPS import GETFPS
from facenet_utils import load_dataset, normalize_vectors, predict_using_classifier

import ctypes
import pyds
import cv2

import os
import os.path
from os import path
import math

MUXER_OUTPUT_WIDTH = 720
MUXER_OUTPUT_HEIGHT = 576
MUXER_BATCH_TIMEOUT_USEC = 400000
TILED_OUTPUT_WIDTH = 720  
TILED_OUTPUT_HEIGHT = 576

fps_stream=None
face_counter= []
frame_count = {}
saved_count = {}

pgie_classes_str = ["Person", "Bag", "Face"]

global PGIE_CLASS_ID_PERSON
PGIE_CLASS_ID_PERSON = 0
PGIE_CLASS_ID_BAG = 1
global PGIE_CLASS_ID_FACE
PGIE_CLASS_ID_FACE = 2

DATASET_PATH = '/home/manjeet/deepstream-facenet/single_face__embeddings.npz'

faces_embeddings, labels = load_dataset(DATASET_PATH)

past_tracking_meta=[0]

# emb_array =[]
# names =[]
# id_name_dic = {}

def osd_sink_pad_buffer_probe(pad,info,u_data):
    # face_thres = 1.0 
    global fps_stream, face_counter, face_count
    frame_number=0
    face_count = 0
    num_rects=0
    
    # with np.load('/home/manjeet/deepstream-face-recognition/app/models/single_face__embeddings.npz') as embeddings:
    #     for key in embeddings.keys():
    #         emb_array.append(embeddings[key])


    # with open("/home/manjeet/deepstream-face-recognition/app/models/names.txt") as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         names.append(line.split()[0])        
        
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
           frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number=frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj=frame_meta.obj_meta_list
        is_first_obj = True
        save_image = False
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            if obj_meta.class_id == PGIE_CLASS_ID_FACE :
                face_count += 1
                if is_first_obj:
                    is_first_obj = False
                    # Getting Image data using nvbufsurface
                    # the input should be address of buffer and batch_id
                    n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                    n_frame = crop_object(n_frame, obj_meta)
                    # convert python array into numpy array format in the copy mode.
                    frame_copy = np.array(n_frame, copy=True, order='C')
                    # convert the array into cv2 default color format
                    frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGRA)


                save_image = True

            # If Tracker is enabled only
            # Retrieve face recognition id for tracked object id
            # Replaces display name for bbox with face recognition id        
            # if str(obj_meta.object_id) in id_name_dic.keys():
            #     obj_meta.text_params.display_text = id_name_dic[str(obj_meta.object_id)]      

            
            l_user = obj_meta.obj_user_meta_list

            # Retrieve TensorMeta data from FaceNet
            # Face recognition is this example is carried out with a simple L2 distance between the embedding from FaceNet
            # with a list of pre-saved embeddings corresponding to the face ids (one in this case)
            if l_user is not None:

                try:
                    # Casting l_user.data to pyds.NvDsUserMeta
                    user_meta=pyds.NvDsUserMeta.cast(l_user.data)
                except StopIteration:
                    break

                if (
                    user_meta.base_meta.meta_type
                    != pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META
                ):
                    continue

                # user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                # v = np.ctypeslib.as_array(ptr, shape=(1,128))
                v = np.ctypeslib.as_array(ptr, shape=(128,))
                
                # # L2 normalization
                # v = v/np.linalg.norm(v)
                # v = v.astype(np.float64)
                
                # for i, emb in enumerate(emb_array):
                #     emb = emb.astype(np.float64)
                #     dist = np.linalg.norm(v - emb)
                #     if dist < face_thres:
                #         id_name_dic[str(obj_meta.object_id)]= names[i]
                #         break
                # Pridict face neme
                yhat = v.reshape((1,-1))
                face_to_predict_embedding = normalize_vectors(yhat)
                result = predict_using_classifier(faces_embeddings, labels, face_to_predict_embedding)
                result =  (str(result).title())
                # print('Predicted name: %s' % result)
                
                # Generate classifer metadata and attach to obj_meta
                
                # Get NvDsClassifierMeta object 
                classifier_meta = pyds.nvds_acquire_classifier_meta_from_pool(batch_meta)

                # Pobulate classifier_meta data with pridction result
                classifier_meta.unique_component_id = tensor_meta.unique_id
                
                
                label_info = pyds.nvds_acquire_label_info_meta_from_pool(batch_meta)

                
                label_info.result_prob = 0
                label_info.result_class_id = 0

                pyds.nvds_add_label_info_meta_to_classifier(classifier_meta, label_info)
                pyds.nvds_add_classifier_meta_to_object(obj_meta, classifier_meta)

                display_text = pyds.get_string(obj_meta.text_params.display_text)
                obj_meta.text_params.display_text = f'{display_text} {result}'
            
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break
        fps_stream.get_fps()
        # Acquiring a display meta object. The memory ownership remains in
        # the C code so downstream plugins can still access it. Otherwise
        # the garbage collector will claim it when this probe function exits.
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_text field here will return the C address of the
        # allocated string. Use pyds.get_string() to get the string content.
        # print("Frame Number=", frame_number, "Number of Objects=", num_rects, "Face_count=",obj_counter[PGIE_CLASS_ID_FACE])
        py_nvosd_text_params.display_text = "Frame Number={} Number of Objects={}  Face Count={}".format(frame_number, num_rects, face_count)
        face_counter.append(face_count)
        #print('face_count',face_count)

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # Using pyds.get_string() to get display_text as string
        print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        if save_image:
            img_path = "{}/stream_{}/frame_{}.jpg".format(folder_name, frame_meta.pad_index, frame_number)
            cv2.imwrite(img_path, frame_copy)
        saved_count["stream_{}".format(frame_meta.pad_index)] += 1

        try:
            l_frame=l_frame.next
        except StopIteration:
            break
			
    return Gst.PadProbeReturn.OK	
def crop_object(image, obj_meta):
    rect_params = obj_meta.rect_params
    top = int(rect_params.top)
    left = int(rect_params.left)
    width = int(rect_params.width)
    height = int(rect_params.height)
    obj_name = pgie_classes_str[obj_meta.class_id]

    crop_img = image[top:top+height, left:left+width]
	
    return crop_img

def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    if (gstname.find("video") != -1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)
    if not is_aarch64() and name.find("nvv4l2decoder") != -1:
        # Use CUDA unified memory in the pipeline so frames
        # can be easily accessed on CPU in Python.
        Object.set_property("cudadec-memtype", 2)
    #    print("Seting bufapi_version\n")
    #    Object.set_property("bufapi-version", True)


def create_source_bin(index, uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-%02d" % index
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin


def main(args):
    global fps_stream, folder_name

    # Check input arguments
    if len(args) < 2:
        sys.stderr.write("usage: %s <uri1> [uri2] ... [uriN] <folder to save frames>\n" % args[0])
        sys.exit(1)

    # Standard GStreamer initialization
    GObject.threads_init()

    number_sources = 0

    global folder_name
    folder_name = args[-1]
    if path.exists(folder_name):
        sys.stderr.write("The output folder %s already exists. Please remove it first.\n" % folder_name)
        sys.exit(1)

    os.mkdir(folder_name)
    print("Frames will be saved in ", folder_name)
    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    print("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")


    pipeline.add(streammux)
    for i in range(number_sources):
        os.mkdir(folder_name + "/stream_" + str(i))
        frame_count["stream_" + str(i)] = 0
        saved_count["stream_" + str(i)] = 0
        print("Creating source_bin ",i," \n ")
        uri_name = 'file://' + sys.argv[1]
        if uri_name.find("rtsp://") == 0 :
            is_live = True
        source_bin=create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname="sink_%u" %i
        sinkpad= streammux.get_request_pad(padname) 
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad=source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)

    # Use nvinfer to run inferencing on camera's output,
    # behaviour of inferencing is set through config file
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write(" Unable to create tracker \n")

    sgie1 = Gst.ElementFactory.make("nvinfer", "secondary1-nvinference-engine")
    if not sgie1:
        sys.stderr.write(" Unable to make sgie1 \n")
        
    # Use convertor to convert from NV12 to RGBA as required by nvosd
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")

    # Create OSD to draw on the converted RGBA buffer
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")

    # # Finally render the osd output
    # if is_aarch64():
    #     transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")

    # print("Creating EGLSink \n")
    # sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    # if not sink:
    #     sys.stderr.write(" Unable to create egl sink \n")
    # sink.set_property('sync', 0)

# RTSP replacement
    nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", "convertor_postosd")
    if not nvvidconv_postosd:
        sys.stderr.write(" Unable to create nvvidconv_postosd \n")
    
    # Create a caps filter
    caps = Gst.ElementFactory.make("capsfilter", "filter")
    caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))
    
    # Make the encoder
    # if codec == "H264":
    encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
    print("Creating H264 Encoder")
    # elif codec == "H265":
    #     encoder = Gst.ElementFactory.make("nvv4l2h265enc", "encoder")
    #     print("Creating H265 Encoder")
    if not encoder:
        sys.stderr.write(" Unable to create encoder")
    encoder.set_property('bitrate', 8000000)
    if is_aarch64():
        encoder.set_property('preset-level', 1)
        encoder.set_property('insert-sps-pps', 1)
        encoder.set_property('bufapi-version', 1)
    
    # Make the payload-encode video into RTP packets
    # if codec == "H264":
    rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
    print("Creating H264 rtppay")
    # elif codec == "H265":
    #     rtppay = Gst.ElementFactory.make("rtph265pay", "rtppay")
    #     print("Creating H265 rtppay")
    if not rtppay:
        sys.stderr.write(" Unable to create rtppay")
    
    # Make the UDP sink
    updsink_port_num = 5400
    sink = Gst.ElementFactory.make("udpsink", "udpsink")
    if not sink:
        sys.stderr.write(" Unable to create udpsink")
    
    sink.set_property('host', '224.224.255.255')
    sink.set_property('port', updsink_port_num)
    sink.set_property('async', False)
    sink.set_property('sync', 0)

####################

    # print("Playing cam %s " %args[1])
    # caps_v4l2src.set_property('caps', Gst.Caps.from_string("video/x-raw, framerate=30/1"))
    # caps_vidconvsrc.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM)"))
    # source.set_property('device', args[1])
    streammux.set_property('width', 1280)
    streammux.set_property('height', 720)
    streammux.set_property('batch-size', 1)
    streammux.set_property('live-source', 1)
    streammux.set_property('batched-push-timeout', 4000000)
    pgie.set_property('config-file-path', "/home/manjeet/deepstream-face-recognition/app/configs/config_infer_primary_peoplenet.txt")
    sgie1.set_property('config-file-path', "/home/manjeet/deepstream-face-recognition/app/configs/classifier_config.txt")

    #Set properties of tracker
    config = configparser.ConfigParser()
    config.read('/home/manjeet/deepstream-face-recognition/app/configs/dstest2_tracker_config.txt')
    config.sections()

    for key in config['tracker']:
        if key == 'tracker-width' :
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height' :
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id' :
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file' :
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file' :
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
        if key == 'enable-batch-process' :
            tracker_enable_batch_process = config.getint('tracker', key)
            tracker.set_property('enable_batch_process', tracker_enable_batch_process)
        if key == 'enable-past-frame' :
            tracker_enable_past_frame = config.getint('tracker', key)
            tracker.set_property('enable_past_frame', tracker_enable_past_frame)
        if key == 'display-tracking-id' :
            tracker_display_tracking_id = config.getint('tracker', key)
            tracker.set_property('display-tracking-id', tracker_display_tracking_id)
            
    print("Adding elements to Pipeline \n")
    # pipeline.add(source)
    # pipeline.add(caps_v4l2src)
    # pipeline.add(vidconvsrc)
    # pipeline.add(nvvidconvsrc)
    # pipeline.add(caps_vidconvsrc)
    # pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(sgie1)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(nvvidconv_postosd)
    pipeline.add(caps)
    pipeline.add(encoder)
    pipeline.add(rtppay)
    pipeline.add(sink)



    print("Linking elements in the Pipeline \n")
    # source.link(caps_v4l2src)
    # caps_v4l2src.link(vidconvsrc)
    # vidconvsrc.link(nvvidconvsrc)
    # nvvidconvsrc.link(caps_vidconvsrc)

    # sinkpad = streammux.get_request_pad("sink_0")
    # if not sinkpad:
    #     sys.stderr.write(" Unable to get the sink pad of streammux \n")
    # srcpad = caps_vidconvsrc.get_static_pad("src")
    # if not srcpad:
    #     sys.stderr.write(" Unable to get source pad of caps_vidconvsrc \n")
    # srcpad.link(sinkpad)

    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(sgie1)
    sgie1.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(nvvidconv_postosd)
    nvvidconv_postosd.link(caps)
    caps.link(encoder)
    encoder.link(rtppay)
    rtppay.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)
    
    # Add RTSP Streaming
    # Start streaming
    rtsp_port_num = 8554
    
    server = GstRtspServer.RTSPServer.new()
    server.props.service = "%d" % rtsp_port_num
    server.attach(None)
    
    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch( "( udpsrc name=pay0 port=%d caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)%s, payload=96 \" )" % (updsink_port_num, "H264"))
    factory.set_shared(True)
    server.get_mount_points().add_factory("/ds-test", factory)

    ######################
    
    # Lets add probe to get informed of the meta data generated, we add probe to
    # the sink pad of the osd element, since by that time, the buffer would have
    # had got all the metadata.
    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")

    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    # start play back and listen to events
    # List the sources
    print("Now playing...")
    for i, source in enumerate(args[:-1]):
        if i != 0:
            print(i, ": ", source)

    print("Starting pipeline \n")
    # start play back and listed to events		
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)

# def parse_args():
#     parser = argparse.ArgumentParser(description='Application Help')
#     # parser.add_argument("-i", "--input", default="/dev/video0",
#     #               help="Path to v4l2 device path such as /dev/video0")

#     args = parser.parse_args()

#     global codec
#     global bitrate
#     # global stream_path
#     global past_tracking

#     codec = "H264"
#     bitrate = 8000000
#     # stream_path = args.input
#     past_tracking = 0
#     return 0
    
if __name__ == '__main__':
    # parse_args()
    sys.exit(main(sys.argv))

