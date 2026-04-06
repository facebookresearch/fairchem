import {
  __esm
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/@codemirror/legacy-modes/mode/protobuf.js
function wordRegexp(words) {
  return new RegExp("^((" + words.join(")|(") + "))\\b", "i");
}
function tokenBase(stream) {
  if (stream.eatSpace())
    return null;
  if (stream.match("//")) {
    stream.skipToEnd();
    return "comment";
  }
  if (stream.match(/^[0-9\.+-]/, false)) {
    if (stream.match(/^[+-]?0x[0-9a-fA-F]+/))
      return "number";
    if (stream.match(/^[+-]?\d*\.\d+([EeDd][+-]?\d+)?/))
      return "number";
    if (stream.match(/^[+-]?\d+([EeDd][+-]?\d+)?/))
      return "number";
  }
  if (stream.match(/^"([^"]|(""))*"/)) {
    return "string";
  }
  if (stream.match(/^'([^']|(''))*'/)) {
    return "string";
  }
  if (stream.match(keywords)) {
    return "keyword";
  }
  if (stream.match(identifiers)) {
    return "variable";
  }
  ;
  stream.next();
  return null;
}
var keywordArray, keywords, identifiers, protobuf;
var init_protobuf = __esm({
  "../../node_modules/@codemirror/legacy-modes/mode/protobuf.js"() {
    keywordArray = [
      "package",
      "message",
      "import",
      "syntax",
      "required",
      "optional",
      "repeated",
      "reserved",
      "default",
      "extensions",
      "packed",
      "bool",
      "bytes",
      "double",
      "enum",
      "float",
      "string",
      "int32",
      "int64",
      "uint32",
      "uint64",
      "sint32",
      "sint64",
      "fixed32",
      "fixed64",
      "sfixed32",
      "sfixed64",
      "option",
      "service",
      "rpc",
      "returns"
    ];
    keywords = wordRegexp(keywordArray);
    identifiers = new RegExp("^[_A-Za-z\xA1-\uFFFF][_A-Za-z0-9\xA1-\uFFFF]*");
    protobuf = {
      name: "protobuf",
      token: tokenBase,
      languageData: {
        autocomplete: keywordArray
      }
    };
  }
});
init_protobuf();
export {
  protobuf
};
//# sourceMappingURL=/build/_shared/protobuf-6EKLLOZO.js.map
