import {
  Map_default,
  baseGetTag_default,
  getNative_default,
  init_Map,
  init_baseGetTag,
  init_getNative,
  init_isArguments,
  init_isArray,
  init_isArrayLike,
  init_isBuffer,
  init_isPrototype,
  init_isTypedArray,
  init_overArg,
  init_root,
  init_toSource,
  isArguments_default,
  isArrayLike_default,
  isArray_default,
  isBuffer_default,
  isPrototype_default,
  isTypedArray_default,
  overArg_default,
  root_default,
  toSource_default
} from "/build/_shared/chunk-T7QTAV5N.js";
import {
  __esm
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/lodash-es/_nativeKeys.js
var nativeKeys, nativeKeys_default;
var init_nativeKeys = __esm({
  "../../node_modules/lodash-es/_nativeKeys.js"() {
    init_overArg();
    nativeKeys = overArg_default(Object.keys, Object);
    nativeKeys_default = nativeKeys;
  }
});

// ../../node_modules/lodash-es/_baseKeys.js
function baseKeys(object) {
  if (!isPrototype_default(object)) {
    return nativeKeys_default(object);
  }
  var result = [];
  for (var key in Object(object)) {
    if (hasOwnProperty.call(object, key) && key != "constructor") {
      result.push(key);
    }
  }
  return result;
}
var objectProto, hasOwnProperty, baseKeys_default;
var init_baseKeys = __esm({
  "../../node_modules/lodash-es/_baseKeys.js"() {
    init_isPrototype();
    init_nativeKeys();
    objectProto = Object.prototype;
    hasOwnProperty = objectProto.hasOwnProperty;
    baseKeys_default = baseKeys;
  }
});

// ../../node_modules/lodash-es/_DataView.js
var DataView, DataView_default;
var init_DataView = __esm({
  "../../node_modules/lodash-es/_DataView.js"() {
    init_getNative();
    init_root();
    DataView = getNative_default(root_default, "DataView");
    DataView_default = DataView;
  }
});

// ../../node_modules/lodash-es/_Promise.js
var Promise2, Promise_default;
var init_Promise = __esm({
  "../../node_modules/lodash-es/_Promise.js"() {
    init_getNative();
    init_root();
    Promise2 = getNative_default(root_default, "Promise");
    Promise_default = Promise2;
  }
});

// ../../node_modules/lodash-es/_Set.js
var Set, Set_default;
var init_Set = __esm({
  "../../node_modules/lodash-es/_Set.js"() {
    init_getNative();
    init_root();
    Set = getNative_default(root_default, "Set");
    Set_default = Set;
  }
});

// ../../node_modules/lodash-es/_WeakMap.js
var WeakMap, WeakMap_default;
var init_WeakMap = __esm({
  "../../node_modules/lodash-es/_WeakMap.js"() {
    init_getNative();
    init_root();
    WeakMap = getNative_default(root_default, "WeakMap");
    WeakMap_default = WeakMap;
  }
});

// ../../node_modules/lodash-es/_getTag.js
var mapTag, objectTag, promiseTag, setTag, weakMapTag, dataViewTag, dataViewCtorString, mapCtorString, promiseCtorString, setCtorString, weakMapCtorString, getTag, getTag_default;
var init_getTag = __esm({
  "../../node_modules/lodash-es/_getTag.js"() {
    init_DataView();
    init_Map();
    init_Promise();
    init_Set();
    init_WeakMap();
    init_baseGetTag();
    init_toSource();
    mapTag = "[object Map]";
    objectTag = "[object Object]";
    promiseTag = "[object Promise]";
    setTag = "[object Set]";
    weakMapTag = "[object WeakMap]";
    dataViewTag = "[object DataView]";
    dataViewCtorString = toSource_default(DataView_default);
    mapCtorString = toSource_default(Map_default);
    promiseCtorString = toSource_default(Promise_default);
    setCtorString = toSource_default(Set_default);
    weakMapCtorString = toSource_default(WeakMap_default);
    getTag = baseGetTag_default;
    if (DataView_default && getTag(new DataView_default(new ArrayBuffer(1))) != dataViewTag || Map_default && getTag(new Map_default()) != mapTag || Promise_default && getTag(Promise_default.resolve()) != promiseTag || Set_default && getTag(new Set_default()) != setTag || WeakMap_default && getTag(new WeakMap_default()) != weakMapTag) {
      getTag = function(value) {
        var result = baseGetTag_default(value), Ctor = result == objectTag ? value.constructor : void 0, ctorString = Ctor ? toSource_default(Ctor) : "";
        if (ctorString) {
          switch (ctorString) {
            case dataViewCtorString:
              return dataViewTag;
            case mapCtorString:
              return mapTag;
            case promiseCtorString:
              return promiseTag;
            case setCtorString:
              return setTag;
            case weakMapCtorString:
              return weakMapTag;
          }
        }
        return result;
      };
    }
    getTag_default = getTag;
  }
});

// ../../node_modules/lodash-es/isEmpty.js
function isEmpty(value) {
  if (value == null) {
    return true;
  }
  if (isArrayLike_default(value) && (isArray_default(value) || typeof value == "string" || typeof value.splice == "function" || isBuffer_default(value) || isTypedArray_default(value) || isArguments_default(value))) {
    return !value.length;
  }
  var tag = getTag_default(value);
  if (tag == mapTag2 || tag == setTag2) {
    return !value.size;
  }
  if (isPrototype_default(value)) {
    return !baseKeys_default(value).length;
  }
  for (var key in value) {
    if (hasOwnProperty2.call(value, key)) {
      return false;
    }
  }
  return true;
}
var mapTag2, setTag2, objectProto2, hasOwnProperty2, isEmpty_default;
var init_isEmpty = __esm({
  "../../node_modules/lodash-es/isEmpty.js"() {
    init_baseKeys();
    init_getTag();
    init_isArguments();
    init_isArray();
    init_isArrayLike();
    init_isBuffer();
    init_isPrototype();
    init_isTypedArray();
    mapTag2 = "[object Map]";
    setTag2 = "[object Set]";
    objectProto2 = Object.prototype;
    hasOwnProperty2 = objectProto2.hasOwnProperty;
    isEmpty_default = isEmpty;
  }
});

export {
  baseKeys_default,
  init_baseKeys,
  Set_default,
  init_Set,
  getTag_default,
  init_getTag,
  isEmpty_default,
  init_isEmpty
};
//# sourceMappingURL=/build/_shared/chunk-SP2MKLPW.js.map
