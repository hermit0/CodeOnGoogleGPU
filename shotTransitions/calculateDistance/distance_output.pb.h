// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: distance_output.proto

#ifndef PROTOBUF_distance_5foutput_2eproto__INCLUDED
#define PROTOBUF_distance_5foutput_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 2006000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 2006001 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)

namespace distance_output {

// Internal implementation detail -- do not call these.
void  protobuf_AddDesc_distance_5foutput_2eproto();
void protobuf_AssignDesc_distance_5foutput_2eproto();
void protobuf_ShutdownFile_distance_5foutput_2eproto();

class video_sequence;
class frame;

// ===================================================================

class video_sequence : public ::google::protobuf::Message {
 public:
  video_sequence();
  virtual ~video_sequence();

  video_sequence(const video_sequence& from);

  inline video_sequence& operator=(const video_sequence& from) {
    CopyFrom(from);
    return *this;
  }

  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _unknown_fields_;
  }

  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return &_unknown_fields_;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const video_sequence& default_instance();

  void Swap(video_sequence* other);

  // implements Message ----------------------------------------------

  video_sequence* New() const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const video_sequence& from);
  void MergeFrom(const video_sequence& from);
  void Clear();
  bool IsInitialized() const;

  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  public:
  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated .distance_output.frame per_frame = 1;
  inline int per_frame_size() const;
  inline void clear_per_frame();
  static const int kPerFrameFieldNumber = 1;
  inline const ::distance_output::frame& per_frame(int index) const;
  inline ::distance_output::frame* mutable_per_frame(int index);
  inline ::distance_output::frame* add_per_frame();
  inline const ::google::protobuf::RepeatedPtrField< ::distance_output::frame >&
      per_frame() const;
  inline ::google::protobuf::RepeatedPtrField< ::distance_output::frame >*
      mutable_per_frame();

  // @@protoc_insertion_point(class_scope:distance_output.video_sequence)
 private:

  ::google::protobuf::UnknownFieldSet _unknown_fields_;

  ::google::protobuf::uint32 _has_bits_[1];
  mutable int _cached_size_;
  ::google::protobuf::RepeatedPtrField< ::distance_output::frame > per_frame_;
  friend void  protobuf_AddDesc_distance_5foutput_2eproto();
  friend void protobuf_AssignDesc_distance_5foutput_2eproto();
  friend void protobuf_ShutdownFile_distance_5foutput_2eproto();

  void InitAsDefaultInstance();
  static video_sequence* default_instance_;
};
// -------------------------------------------------------------------

class frame : public ::google::protobuf::Message {
 public:
  frame();
  virtual ~frame();

  frame(const frame& from);

  inline frame& operator=(const frame& from) {
    CopyFrom(from);
    return *this;
  }

  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _unknown_fields_;
  }

  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return &_unknown_fields_;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const frame& default_instance();

  void Swap(frame* other);

  // implements Message ----------------------------------------------

  frame* New() const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const frame& from);
  void MergeFrom(const frame& from);
  void Clear();
  bool IsInitialized() const;

  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  public:
  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // required int32 frame_no = 1;
  inline bool has_frame_no() const;
  inline void clear_frame_no();
  static const int kFrameNoFieldNumber = 1;
  inline ::google::protobuf::int32 frame_no() const;
  inline void set_frame_no(::google::protobuf::int32 value);

  // required float value = 2;
  inline bool has_value() const;
  inline void clear_value();
  static const int kValueFieldNumber = 2;
  inline float value() const;
  inline void set_value(float value);

  // @@protoc_insertion_point(class_scope:distance_output.frame)
 private:
  inline void set_has_frame_no();
  inline void clear_has_frame_no();
  inline void set_has_value();
  inline void clear_has_value();

  ::google::protobuf::UnknownFieldSet _unknown_fields_;

  ::google::protobuf::uint32 _has_bits_[1];
  mutable int _cached_size_;
  ::google::protobuf::int32 frame_no_;
  float value_;
  friend void  protobuf_AddDesc_distance_5foutput_2eproto();
  friend void protobuf_AssignDesc_distance_5foutput_2eproto();
  friend void protobuf_ShutdownFile_distance_5foutput_2eproto();

  void InitAsDefaultInstance();
  static frame* default_instance_;
};
// ===================================================================


// ===================================================================

// video_sequence

// repeated .distance_output.frame per_frame = 1;
inline int video_sequence::per_frame_size() const {
  return per_frame_.size();
}
inline void video_sequence::clear_per_frame() {
  per_frame_.Clear();
}
inline const ::distance_output::frame& video_sequence::per_frame(int index) const {
  // @@protoc_insertion_point(field_get:distance_output.video_sequence.per_frame)
  return per_frame_.Get(index);
}
inline ::distance_output::frame* video_sequence::mutable_per_frame(int index) {
  // @@protoc_insertion_point(field_mutable:distance_output.video_sequence.per_frame)
  return per_frame_.Mutable(index);
}
inline ::distance_output::frame* video_sequence::add_per_frame() {
  // @@protoc_insertion_point(field_add:distance_output.video_sequence.per_frame)
  return per_frame_.Add();
}
inline const ::google::protobuf::RepeatedPtrField< ::distance_output::frame >&
video_sequence::per_frame() const {
  // @@protoc_insertion_point(field_list:distance_output.video_sequence.per_frame)
  return per_frame_;
}
inline ::google::protobuf::RepeatedPtrField< ::distance_output::frame >*
video_sequence::mutable_per_frame() {
  // @@protoc_insertion_point(field_mutable_list:distance_output.video_sequence.per_frame)
  return &per_frame_;
}

// -------------------------------------------------------------------

// frame

// required int32 frame_no = 1;
inline bool frame::has_frame_no() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void frame::set_has_frame_no() {
  _has_bits_[0] |= 0x00000001u;
}
inline void frame::clear_has_frame_no() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void frame::clear_frame_no() {
  frame_no_ = 0;
  clear_has_frame_no();
}
inline ::google::protobuf::int32 frame::frame_no() const {
  // @@protoc_insertion_point(field_get:distance_output.frame.frame_no)
  return frame_no_;
}
inline void frame::set_frame_no(::google::protobuf::int32 value) {
  set_has_frame_no();
  frame_no_ = value;
  // @@protoc_insertion_point(field_set:distance_output.frame.frame_no)
}

// required float value = 2;
inline bool frame::has_value() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void frame::set_has_value() {
  _has_bits_[0] |= 0x00000002u;
}
inline void frame::clear_has_value() {
  _has_bits_[0] &= ~0x00000002u;
}
inline void frame::clear_value() {
  value_ = 0;
  clear_has_value();
}
inline float frame::value() const {
  // @@protoc_insertion_point(field_get:distance_output.frame.value)
  return value_;
}
inline void frame::set_value(float value) {
  set_has_value();
  value_ = value;
  // @@protoc_insertion_point(field_set:distance_output.frame.value)
}


// @@protoc_insertion_point(namespace_scope)

}  // namespace distance_output

#ifndef SWIG
namespace google {
namespace protobuf {


}  // namespace google
}  // namespace protobuf
#endif  // SWIG

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_distance_5foutput_2eproto__INCLUDED
