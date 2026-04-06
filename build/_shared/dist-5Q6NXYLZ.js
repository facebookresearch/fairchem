import {
  RuleId,
  fileError,
  fileWarn,
  getMetadataTags,
  selectAll
} from "/build/_shared/chunk-Q6DHUCUI.js";
import "/build/_shared/chunk-LPSXN4QV.js";
import {
  __commonJS,
  __toESM
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/docx/build/index.js
var require_build = __commonJS({
  "../../node_modules/docx/build/index.js"(exports3, module) {
    !function(e, t) {
      "object" == typeof exports3 && "object" == typeof module ? module.exports = t() : "function" == typeof define && define.amd ? define([], t) : "object" == typeof exports3 ? exports3.docx = t() : e.docx = t();
    }(globalThis, () => (() => {
      var e = { 9742: (e2, t2) => {
        "use strict";
        t2.byteLength = function(e3) {
          var t3 = c(e3), r3 = t3[0], n3 = t3[1];
          return 3 * (r3 + n3) / 4 - n3;
        }, t2.toByteArray = function(e3) {
          var t3, r3, i2 = c(e3), o2 = i2[0], a2 = i2[1], u2 = new s(function(e4, t4, r4) {
            return 3 * (t4 + r4) / 4 - r4;
          }(0, o2, a2)), l = 0, h = a2 > 0 ? o2 - 4 : o2;
          for (r3 = 0; r3 < h; r3 += 4)
            t3 = n2[e3.charCodeAt(r3)] << 18 | n2[e3.charCodeAt(r3 + 1)] << 12 | n2[e3.charCodeAt(r3 + 2)] << 6 | n2[e3.charCodeAt(r3 + 3)], u2[l++] = t3 >> 16 & 255, u2[l++] = t3 >> 8 & 255, u2[l++] = 255 & t3;
          return 2 === a2 && (t3 = n2[e3.charCodeAt(r3)] << 2 | n2[e3.charCodeAt(r3 + 1)] >> 4, u2[l++] = 255 & t3), 1 === a2 && (t3 = n2[e3.charCodeAt(r3)] << 10 | n2[e3.charCodeAt(r3 + 1)] << 4 | n2[e3.charCodeAt(r3 + 2)] >> 2, u2[l++] = t3 >> 8 & 255, u2[l++] = 255 & t3), u2;
        }, t2.fromByteArray = function(e3) {
          for (var t3, n3 = e3.length, s2 = n3 % 3, i2 = [], o2 = 16383, a2 = 0, c2 = n3 - s2; a2 < c2; a2 += o2)
            i2.push(u(e3, a2, a2 + o2 > c2 ? c2 : a2 + o2));
          return 1 === s2 ? (t3 = e3[n3 - 1], i2.push(r2[t3 >> 2] + r2[t3 << 4 & 63] + "==")) : 2 === s2 && (t3 = (e3[n3 - 2] << 8) + e3[n3 - 1], i2.push(r2[t3 >> 10] + r2[t3 >> 4 & 63] + r2[t3 << 2 & 63] + "=")), i2.join("");
        };
        for (var r2 = [], n2 = [], s = "undefined" != typeof Uint8Array ? Uint8Array : Array, i = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/", o = 0, a = i.length; o < a; ++o)
          r2[o] = i[o], n2[i.charCodeAt(o)] = o;
        function c(e3) {
          var t3 = e3.length;
          if (t3 % 4 > 0)
            throw new Error("Invalid string. Length must be a multiple of 4");
          var r3 = e3.indexOf("=");
          return -1 === r3 && (r3 = t3), [r3, r3 === t3 ? 0 : 4 - r3 % 4];
        }
        function u(e3, t3, n3) {
          for (var s2, i2, o2 = [], a2 = t3; a2 < n3; a2 += 3)
            s2 = (e3[a2] << 16 & 16711680) + (e3[a2 + 1] << 8 & 65280) + (255 & e3[a2 + 2]), o2.push(r2[(i2 = s2) >> 18 & 63] + r2[i2 >> 12 & 63] + r2[i2 >> 6 & 63] + r2[63 & i2]);
          return o2.join("");
        }
        n2["-".charCodeAt(0)] = 62, n2["_".charCodeAt(0)] = 63;
      }, 8764: (e2, t2, r2) => {
        "use strict";
        const n2 = r2(9742), s = r2(645), i = "function" == typeof Symbol && "function" == typeof Symbol.for ? Symbol.for("nodejs.util.inspect.custom") : null;
        t2.Buffer = c, t2.SlowBuffer = function(e3) {
          return +e3 != e3 && (e3 = 0), c.alloc(+e3);
        }, t2.INSPECT_MAX_BYTES = 50;
        const o = 2147483647;
        function a(e3) {
          if (e3 > o)
            throw new RangeError('The value "' + e3 + '" is invalid for option "size"');
          const t3 = new Uint8Array(e3);
          return Object.setPrototypeOf(t3, c.prototype), t3;
        }
        function c(e3, t3, r3) {
          if ("number" == typeof e3) {
            if ("string" == typeof t3)
              throw new TypeError('The "string" argument must be of type string. Received type number');
            return h(e3);
          }
          return u(e3, t3, r3);
        }
        function u(e3, t3, r3) {
          if ("string" == typeof e3)
            return function(e4, t4) {
              if ("string" == typeof t4 && "" !== t4 || (t4 = "utf8"), !c.isEncoding(t4))
                throw new TypeError("Unknown encoding: " + t4);
              const r4 = 0 | m(e4, t4);
              let n4 = a(r4);
              const s3 = n4.write(e4, t4);
              return s3 !== r4 && (n4 = n4.slice(0, s3)), n4;
            }(e3, t3);
          if (ArrayBuffer.isView(e3))
            return function(e4) {
              if (q(e4, Uint8Array)) {
                const t4 = new Uint8Array(e4);
                return d(t4.buffer, t4.byteOffset, t4.byteLength);
              }
              return p(e4);
            }(e3);
          if (null == e3)
            throw new TypeError("The first argument must be one of type string, Buffer, ArrayBuffer, Array, or Array-like Object. Received type " + typeof e3);
          if (q(e3, ArrayBuffer) || e3 && q(e3.buffer, ArrayBuffer))
            return d(e3, t3, r3);
          if ("undefined" != typeof SharedArrayBuffer && (q(e3, SharedArrayBuffer) || e3 && q(e3.buffer, SharedArrayBuffer)))
            return d(e3, t3, r3);
          if ("number" == typeof e3)
            throw new TypeError('The "value" argument must not be of type number. Received type number');
          const n3 = e3.valueOf && e3.valueOf();
          if (null != n3 && n3 !== e3)
            return c.from(n3, t3, r3);
          const s2 = function(e4) {
            if (c.isBuffer(e4)) {
              const t4 = 0 | f(e4.length), r4 = a(t4);
              return 0 === r4.length || e4.copy(r4, 0, 0, t4), r4;
            }
            return void 0 !== e4.length ? "number" != typeof e4.length || Z(e4.length) ? a(0) : p(e4) : "Buffer" === e4.type && Array.isArray(e4.data) ? p(e4.data) : void 0;
          }(e3);
          if (s2)
            return s2;
          if ("undefined" != typeof Symbol && null != Symbol.toPrimitive && "function" == typeof e3[Symbol.toPrimitive])
            return c.from(e3[Symbol.toPrimitive]("string"), t3, r3);
          throw new TypeError("The first argument must be one of type string, Buffer, ArrayBuffer, Array, or Array-like Object. Received type " + typeof e3);
        }
        function l(e3) {
          if ("number" != typeof e3)
            throw new TypeError('"size" argument must be of type number');
          if (e3 < 0)
            throw new RangeError('The value "' + e3 + '" is invalid for option "size"');
        }
        function h(e3) {
          return l(e3), a(e3 < 0 ? 0 : 0 | f(e3));
        }
        function p(e3) {
          const t3 = e3.length < 0 ? 0 : 0 | f(e3.length), r3 = a(t3);
          for (let n3 = 0; n3 < t3; n3 += 1)
            r3[n3] = 255 & e3[n3];
          return r3;
        }
        function d(e3, t3, r3) {
          if (t3 < 0 || e3.byteLength < t3)
            throw new RangeError('"offset" is outside of buffer bounds');
          if (e3.byteLength < t3 + (r3 || 0))
            throw new RangeError('"length" is outside of buffer bounds');
          let n3;
          return n3 = void 0 === t3 && void 0 === r3 ? new Uint8Array(e3) : void 0 === r3 ? new Uint8Array(e3, t3) : new Uint8Array(e3, t3, r3), Object.setPrototypeOf(n3, c.prototype), n3;
        }
        function f(e3) {
          if (e3 >= o)
            throw new RangeError("Attempt to allocate Buffer larger than maximum size: 0x" + o.toString(16) + " bytes");
          return 0 | e3;
        }
        function m(e3, t3) {
          if (c.isBuffer(e3))
            return e3.length;
          if (ArrayBuffer.isView(e3) || q(e3, ArrayBuffer))
            return e3.byteLength;
          if ("string" != typeof e3)
            throw new TypeError('The "string" argument must be one of type string, Buffer, or ArrayBuffer. Received type ' + typeof e3);
          const r3 = e3.length, n3 = arguments.length > 2 && true === arguments[2];
          if (!n3 && 0 === r3)
            return 0;
          let s2 = false;
          for (; ; )
            switch (t3) {
              case "ascii":
              case "latin1":
              case "binary":
                return r3;
              case "utf8":
              case "utf-8":
                return V(e3).length;
              case "ucs2":
              case "ucs-2":
              case "utf16le":
              case "utf-16le":
                return 2 * r3;
              case "hex":
                return r3 >>> 1;
              case "base64":
                return $(e3).length;
              default:
                if (s2)
                  return n3 ? -1 : V(e3).length;
                t3 = ("" + t3).toLowerCase(), s2 = true;
            }
        }
        function w(e3, t3, r3) {
          let n3 = false;
          if ((void 0 === t3 || t3 < 0) && (t3 = 0), t3 > this.length)
            return "";
          if ((void 0 === r3 || r3 > this.length) && (r3 = this.length), r3 <= 0)
            return "";
          if ((r3 >>>= 0) <= (t3 >>>= 0))
            return "";
          for (e3 || (e3 = "utf8"); ; )
            switch (e3) {
              case "hex":
                return C(this, t3, r3);
              case "utf8":
              case "utf-8":
                return S(this, t3, r3);
              case "ascii":
                return R(this, t3, r3);
              case "latin1":
              case "binary":
                return N(this, t3, r3);
              case "base64":
                return A(this, t3, r3);
              case "ucs2":
              case "ucs-2":
              case "utf16le":
              case "utf-16le":
                return O(this, t3, r3);
              default:
                if (n3)
                  throw new TypeError("Unknown encoding: " + e3);
                e3 = (e3 + "").toLowerCase(), n3 = true;
            }
        }
        function g(e3, t3, r3) {
          const n3 = e3[t3];
          e3[t3] = e3[r3], e3[r3] = n3;
        }
        function y(e3, t3, r3, n3, s2) {
          if (0 === e3.length)
            return -1;
          if ("string" == typeof r3 ? (n3 = r3, r3 = 0) : r3 > 2147483647 ? r3 = 2147483647 : r3 < -2147483648 && (r3 = -2147483648), Z(r3 = +r3) && (r3 = s2 ? 0 : e3.length - 1), r3 < 0 && (r3 = e3.length + r3), r3 >= e3.length) {
            if (s2)
              return -1;
            r3 = e3.length - 1;
          } else if (r3 < 0) {
            if (!s2)
              return -1;
            r3 = 0;
          }
          if ("string" == typeof t3 && (t3 = c.from(t3, n3)), c.isBuffer(t3))
            return 0 === t3.length ? -1 : b(e3, t3, r3, n3, s2);
          if ("number" == typeof t3)
            return t3 &= 255, "function" == typeof Uint8Array.prototype.indexOf ? s2 ? Uint8Array.prototype.indexOf.call(e3, t3, r3) : Uint8Array.prototype.lastIndexOf.call(e3, t3, r3) : b(e3, [t3], r3, n3, s2);
          throw new TypeError("val must be string, number or Buffer");
        }
        function b(e3, t3, r3, n3, s2) {
          let i2, o2 = 1, a2 = e3.length, c2 = t3.length;
          if (void 0 !== n3 && ("ucs2" === (n3 = String(n3).toLowerCase()) || "ucs-2" === n3 || "utf16le" === n3 || "utf-16le" === n3)) {
            if (e3.length < 2 || t3.length < 2)
              return -1;
            o2 = 2, a2 /= 2, c2 /= 2, r3 /= 2;
          }
          function u2(e4, t4) {
            return 1 === o2 ? e4[t4] : e4.readUInt16BE(t4 * o2);
          }
          if (s2) {
            let n4 = -1;
            for (i2 = r3; i2 < a2; i2++)
              if (u2(e3, i2) === u2(t3, -1 === n4 ? 0 : i2 - n4)) {
                if (-1 === n4 && (n4 = i2), i2 - n4 + 1 === c2)
                  return n4 * o2;
              } else
                -1 !== n4 && (i2 -= i2 - n4), n4 = -1;
          } else
            for (r3 + c2 > a2 && (r3 = a2 - c2), i2 = r3; i2 >= 0; i2--) {
              let r4 = true;
              for (let n4 = 0; n4 < c2; n4++)
                if (u2(e3, i2 + n4) !== u2(t3, n4)) {
                  r4 = false;
                  break;
                }
              if (r4)
                return i2;
            }
          return -1;
        }
        function x(e3, t3, r3, n3) {
          r3 = Number(r3) || 0;
          const s2 = e3.length - r3;
          n3 ? (n3 = Number(n3)) > s2 && (n3 = s2) : n3 = s2;
          const i2 = t3.length;
          let o2;
          for (n3 > i2 / 2 && (n3 = i2 / 2), o2 = 0; o2 < n3; ++o2) {
            const n4 = parseInt(t3.substr(2 * o2, 2), 16);
            if (Z(n4))
              return o2;
            e3[r3 + o2] = n4;
          }
          return o2;
        }
        function v(e3, t3, r3, n3) {
          return X(V(t3, e3.length - r3), e3, r3, n3);
        }
        function _(e3, t3, r3, n3) {
          return X(function(e4) {
            const t4 = [];
            for (let r4 = 0; r4 < e4.length; ++r4)
              t4.push(255 & e4.charCodeAt(r4));
            return t4;
          }(t3), e3, r3, n3);
        }
        function E(e3, t3, r3, n3) {
          return X($(t3), e3, r3, n3);
        }
        function T(e3, t3, r3, n3) {
          return X(function(e4, t4) {
            let r4, n4, s2;
            const i2 = [];
            for (let o2 = 0; o2 < e4.length && !((t4 -= 2) < 0); ++o2)
              r4 = e4.charCodeAt(o2), n4 = r4 >> 8, s2 = r4 % 256, i2.push(s2), i2.push(n4);
            return i2;
          }(t3, e3.length - r3), e3, r3, n3);
        }
        function A(e3, t3, r3) {
          return 0 === t3 && r3 === e3.length ? n2.fromByteArray(e3) : n2.fromByteArray(e3.slice(t3, r3));
        }
        function S(e3, t3, r3) {
          r3 = Math.min(e3.length, r3);
          const n3 = [];
          let s2 = t3;
          for (; s2 < r3; ) {
            const t4 = e3[s2];
            let i2 = null, o2 = t4 > 239 ? 4 : t4 > 223 ? 3 : t4 > 191 ? 2 : 1;
            if (s2 + o2 <= r3) {
              let r4, n4, a2, c2;
              switch (o2) {
                case 1:
                  t4 < 128 && (i2 = t4);
                  break;
                case 2:
                  r4 = e3[s2 + 1], 128 == (192 & r4) && (c2 = (31 & t4) << 6 | 63 & r4, c2 > 127 && (i2 = c2));
                  break;
                case 3:
                  r4 = e3[s2 + 1], n4 = e3[s2 + 2], 128 == (192 & r4) && 128 == (192 & n4) && (c2 = (15 & t4) << 12 | (63 & r4) << 6 | 63 & n4, c2 > 2047 && (c2 < 55296 || c2 > 57343) && (i2 = c2));
                  break;
                case 4:
                  r4 = e3[s2 + 1], n4 = e3[s2 + 2], a2 = e3[s2 + 3], 128 == (192 & r4) && 128 == (192 & n4) && 128 == (192 & a2) && (c2 = (15 & t4) << 18 | (63 & r4) << 12 | (63 & n4) << 6 | 63 & a2, c2 > 65535 && c2 < 1114112 && (i2 = c2));
              }
            }
            null === i2 ? (i2 = 65533, o2 = 1) : i2 > 65535 && (i2 -= 65536, n3.push(i2 >>> 10 & 1023 | 55296), i2 = 56320 | 1023 & i2), n3.push(i2), s2 += o2;
          }
          return function(e4) {
            const t4 = e4.length;
            if (t4 <= I)
              return String.fromCharCode.apply(String, e4);
            let r4 = "", n4 = 0;
            for (; n4 < t4; )
              r4 += String.fromCharCode.apply(String, e4.slice(n4, n4 += I));
            return r4;
          }(n3);
        }
        t2.kMaxLength = o, c.TYPED_ARRAY_SUPPORT = function() {
          try {
            const e3 = new Uint8Array(1), t3 = { foo: function() {
              return 42;
            } };
            return Object.setPrototypeOf(t3, Uint8Array.prototype), Object.setPrototypeOf(e3, t3), 42 === e3.foo();
          } catch (e3) {
            return false;
          }
        }(), c.TYPED_ARRAY_SUPPORT || "undefined" == typeof console || "function" != typeof console.error || console.error("This browser lacks typed array (Uint8Array) support which is required by `buffer` v5.x. Use `buffer` v4.x if you require old browser support."), Object.defineProperty(c.prototype, "parent", { enumerable: true, get: function() {
          if (c.isBuffer(this))
            return this.buffer;
        } }), Object.defineProperty(c.prototype, "offset", { enumerable: true, get: function() {
          if (c.isBuffer(this))
            return this.byteOffset;
        } }), c.poolSize = 8192, c.from = function(e3, t3, r3) {
          return u(e3, t3, r3);
        }, Object.setPrototypeOf(c.prototype, Uint8Array.prototype), Object.setPrototypeOf(c, Uint8Array), c.alloc = function(e3, t3, r3) {
          return function(e4, t4, r4) {
            return l(e4), e4 <= 0 ? a(e4) : void 0 !== t4 ? "string" == typeof r4 ? a(e4).fill(t4, r4) : a(e4).fill(t4) : a(e4);
          }(e3, t3, r3);
        }, c.allocUnsafe = function(e3) {
          return h(e3);
        }, c.allocUnsafeSlow = function(e3) {
          return h(e3);
        }, c.isBuffer = function(e3) {
          return null != e3 && true === e3._isBuffer && e3 !== c.prototype;
        }, c.compare = function(e3, t3) {
          if (q(e3, Uint8Array) && (e3 = c.from(e3, e3.offset, e3.byteLength)), q(t3, Uint8Array) && (t3 = c.from(t3, t3.offset, t3.byteLength)), !c.isBuffer(e3) || !c.isBuffer(t3))
            throw new TypeError('The "buf1", "buf2" arguments must be one of type Buffer or Uint8Array');
          if (e3 === t3)
            return 0;
          let r3 = e3.length, n3 = t3.length;
          for (let s2 = 0, i2 = Math.min(r3, n3); s2 < i2; ++s2)
            if (e3[s2] !== t3[s2]) {
              r3 = e3[s2], n3 = t3[s2];
              break;
            }
          return r3 < n3 ? -1 : n3 < r3 ? 1 : 0;
        }, c.isEncoding = function(e3) {
          switch (String(e3).toLowerCase()) {
            case "hex":
            case "utf8":
            case "utf-8":
            case "ascii":
            case "latin1":
            case "binary":
            case "base64":
            case "ucs2":
            case "ucs-2":
            case "utf16le":
            case "utf-16le":
              return true;
            default:
              return false;
          }
        }, c.concat = function(e3, t3) {
          if (!Array.isArray(e3))
            throw new TypeError('"list" argument must be an Array of Buffers');
          if (0 === e3.length)
            return c.alloc(0);
          let r3;
          if (void 0 === t3)
            for (t3 = 0, r3 = 0; r3 < e3.length; ++r3)
              t3 += e3[r3].length;
          const n3 = c.allocUnsafe(t3);
          let s2 = 0;
          for (r3 = 0; r3 < e3.length; ++r3) {
            let t4 = e3[r3];
            if (q(t4, Uint8Array))
              s2 + t4.length > n3.length ? (c.isBuffer(t4) || (t4 = c.from(t4)), t4.copy(n3, s2)) : Uint8Array.prototype.set.call(n3, t4, s2);
            else {
              if (!c.isBuffer(t4))
                throw new TypeError('"list" argument must be an Array of Buffers');
              t4.copy(n3, s2);
            }
            s2 += t4.length;
          }
          return n3;
        }, c.byteLength = m, c.prototype._isBuffer = true, c.prototype.swap16 = function() {
          const e3 = this.length;
          if (e3 % 2 != 0)
            throw new RangeError("Buffer size must be a multiple of 16-bits");
          for (let t3 = 0; t3 < e3; t3 += 2)
            g(this, t3, t3 + 1);
          return this;
        }, c.prototype.swap32 = function() {
          const e3 = this.length;
          if (e3 % 4 != 0)
            throw new RangeError("Buffer size must be a multiple of 32-bits");
          for (let t3 = 0; t3 < e3; t3 += 4)
            g(this, t3, t3 + 3), g(this, t3 + 1, t3 + 2);
          return this;
        }, c.prototype.swap64 = function() {
          const e3 = this.length;
          if (e3 % 8 != 0)
            throw new RangeError("Buffer size must be a multiple of 64-bits");
          for (let t3 = 0; t3 < e3; t3 += 8)
            g(this, t3, t3 + 7), g(this, t3 + 1, t3 + 6), g(this, t3 + 2, t3 + 5), g(this, t3 + 3, t3 + 4);
          return this;
        }, c.prototype.toString = function() {
          const e3 = this.length;
          return 0 === e3 ? "" : 0 === arguments.length ? S(this, 0, e3) : w.apply(this, arguments);
        }, c.prototype.toLocaleString = c.prototype.toString, c.prototype.equals = function(e3) {
          if (!c.isBuffer(e3))
            throw new TypeError("Argument must be a Buffer");
          return this === e3 || 0 === c.compare(this, e3);
        }, c.prototype.inspect = function() {
          let e3 = "";
          const r3 = t2.INSPECT_MAX_BYTES;
          return e3 = this.toString("hex", 0, r3).replace(/(.{2})/g, "$1 ").trim(), this.length > r3 && (e3 += " ... "), "<Buffer " + e3 + ">";
        }, i && (c.prototype[i] = c.prototype.inspect), c.prototype.compare = function(e3, t3, r3, n3, s2) {
          if (q(e3, Uint8Array) && (e3 = c.from(e3, e3.offset, e3.byteLength)), !c.isBuffer(e3))
            throw new TypeError('The "target" argument must be one of type Buffer or Uint8Array. Received type ' + typeof e3);
          if (void 0 === t3 && (t3 = 0), void 0 === r3 && (r3 = e3 ? e3.length : 0), void 0 === n3 && (n3 = 0), void 0 === s2 && (s2 = this.length), t3 < 0 || r3 > e3.length || n3 < 0 || s2 > this.length)
            throw new RangeError("out of range index");
          if (n3 >= s2 && t3 >= r3)
            return 0;
          if (n3 >= s2)
            return -1;
          if (t3 >= r3)
            return 1;
          if (this === e3)
            return 0;
          let i2 = (s2 >>>= 0) - (n3 >>>= 0), o2 = (r3 >>>= 0) - (t3 >>>= 0);
          const a2 = Math.min(i2, o2), u2 = this.slice(n3, s2), l2 = e3.slice(t3, r3);
          for (let e4 = 0; e4 < a2; ++e4)
            if (u2[e4] !== l2[e4]) {
              i2 = u2[e4], o2 = l2[e4];
              break;
            }
          return i2 < o2 ? -1 : o2 < i2 ? 1 : 0;
        }, c.prototype.includes = function(e3, t3, r3) {
          return -1 !== this.indexOf(e3, t3, r3);
        }, c.prototype.indexOf = function(e3, t3, r3) {
          return y(this, e3, t3, r3, true);
        }, c.prototype.lastIndexOf = function(e3, t3, r3) {
          return y(this, e3, t3, r3, false);
        }, c.prototype.write = function(e3, t3, r3, n3) {
          if (void 0 === t3)
            n3 = "utf8", r3 = this.length, t3 = 0;
          else if (void 0 === r3 && "string" == typeof t3)
            n3 = t3, r3 = this.length, t3 = 0;
          else {
            if (!isFinite(t3))
              throw new Error("Buffer.write(string, encoding, offset[, length]) is no longer supported");
            t3 >>>= 0, isFinite(r3) ? (r3 >>>= 0, void 0 === n3 && (n3 = "utf8")) : (n3 = r3, r3 = void 0);
          }
          const s2 = this.length - t3;
          if ((void 0 === r3 || r3 > s2) && (r3 = s2), e3.length > 0 && (r3 < 0 || t3 < 0) || t3 > this.length)
            throw new RangeError("Attempt to write outside buffer bounds");
          n3 || (n3 = "utf8");
          let i2 = false;
          for (; ; )
            switch (n3) {
              case "hex":
                return x(this, e3, t3, r3);
              case "utf8":
              case "utf-8":
                return v(this, e3, t3, r3);
              case "ascii":
              case "latin1":
              case "binary":
                return _(this, e3, t3, r3);
              case "base64":
                return E(this, e3, t3, r3);
              case "ucs2":
              case "ucs-2":
              case "utf16le":
              case "utf-16le":
                return T(this, e3, t3, r3);
              default:
                if (i2)
                  throw new TypeError("Unknown encoding: " + n3);
                n3 = ("" + n3).toLowerCase(), i2 = true;
            }
        }, c.prototype.toJSON = function() {
          return { type: "Buffer", data: Array.prototype.slice.call(this._arr || this, 0) };
        };
        const I = 4096;
        function R(e3, t3, r3) {
          let n3 = "";
          r3 = Math.min(e3.length, r3);
          for (let s2 = t3; s2 < r3; ++s2)
            n3 += String.fromCharCode(127 & e3[s2]);
          return n3;
        }
        function N(e3, t3, r3) {
          let n3 = "";
          r3 = Math.min(e3.length, r3);
          for (let s2 = t3; s2 < r3; ++s2)
            n3 += String.fromCharCode(e3[s2]);
          return n3;
        }
        function C(e3, t3, r3) {
          const n3 = e3.length;
          (!t3 || t3 < 0) && (t3 = 0), (!r3 || r3 < 0 || r3 > n3) && (r3 = n3);
          let s2 = "";
          for (let n4 = t3; n4 < r3; ++n4)
            s2 += Y[e3[n4]];
          return s2;
        }
        function O(e3, t3, r3) {
          const n3 = e3.slice(t3, r3);
          let s2 = "";
          for (let e4 = 0; e4 < n3.length - 1; e4 += 2)
            s2 += String.fromCharCode(n3[e4] + 256 * n3[e4 + 1]);
          return s2;
        }
        function k(e3, t3, r3) {
          if (e3 % 1 != 0 || e3 < 0)
            throw new RangeError("offset is not uint");
          if (e3 + t3 > r3)
            throw new RangeError("Trying to access beyond buffer length");
        }
        function L(e3, t3, r3, n3, s2, i2) {
          if (!c.isBuffer(e3))
            throw new TypeError('"buffer" argument must be a Buffer instance');
          if (t3 > s2 || t3 < i2)
            throw new RangeError('"value" argument is out of bounds');
          if (r3 + n3 > e3.length)
            throw new RangeError("Index out of range");
        }
        function D(e3, t3, r3, n3, s2) {
          j(t3, n3, s2, e3, r3, 7);
          let i2 = Number(t3 & BigInt(4294967295));
          e3[r3++] = i2, i2 >>= 8, e3[r3++] = i2, i2 >>= 8, e3[r3++] = i2, i2 >>= 8, e3[r3++] = i2;
          let o2 = Number(t3 >> BigInt(32) & BigInt(4294967295));
          return e3[r3++] = o2, o2 >>= 8, e3[r3++] = o2, o2 >>= 8, e3[r3++] = o2, o2 >>= 8, e3[r3++] = o2, r3;
        }
        function P(e3, t3, r3, n3, s2) {
          j(t3, n3, s2, e3, r3, 7);
          let i2 = Number(t3 & BigInt(4294967295));
          e3[r3 + 7] = i2, i2 >>= 8, e3[r3 + 6] = i2, i2 >>= 8, e3[r3 + 5] = i2, i2 >>= 8, e3[r3 + 4] = i2;
          let o2 = Number(t3 >> BigInt(32) & BigInt(4294967295));
          return e3[r3 + 3] = o2, o2 >>= 8, e3[r3 + 2] = o2, o2 >>= 8, e3[r3 + 1] = o2, o2 >>= 8, e3[r3] = o2, r3 + 8;
        }
        function F(e3, t3, r3, n3, s2, i2) {
          if (r3 + n3 > e3.length)
            throw new RangeError("Index out of range");
          if (r3 < 0)
            throw new RangeError("Index out of range");
        }
        function B(e3, t3, r3, n3, i2) {
          return t3 = +t3, r3 >>>= 0, i2 || F(e3, 0, r3, 4), s.write(e3, t3, r3, n3, 23, 4), r3 + 4;
        }
        function M(e3, t3, r3, n3, i2) {
          return t3 = +t3, r3 >>>= 0, i2 || F(e3, 0, r3, 8), s.write(e3, t3, r3, n3, 52, 8), r3 + 8;
        }
        c.prototype.slice = function(e3, t3) {
          const r3 = this.length;
          (e3 = ~~e3) < 0 ? (e3 += r3) < 0 && (e3 = 0) : e3 > r3 && (e3 = r3), (t3 = void 0 === t3 ? r3 : ~~t3) < 0 ? (t3 += r3) < 0 && (t3 = 0) : t3 > r3 && (t3 = r3), t3 < e3 && (t3 = e3);
          const n3 = this.subarray(e3, t3);
          return Object.setPrototypeOf(n3, c.prototype), n3;
        }, c.prototype.readUintLE = c.prototype.readUIntLE = function(e3, t3, r3) {
          e3 >>>= 0, t3 >>>= 0, r3 || k(e3, t3, this.length);
          let n3 = this[e3], s2 = 1, i2 = 0;
          for (; ++i2 < t3 && (s2 *= 256); )
            n3 += this[e3 + i2] * s2;
          return n3;
        }, c.prototype.readUintBE = c.prototype.readUIntBE = function(e3, t3, r3) {
          e3 >>>= 0, t3 >>>= 0, r3 || k(e3, t3, this.length);
          let n3 = this[e3 + --t3], s2 = 1;
          for (; t3 > 0 && (s2 *= 256); )
            n3 += this[e3 + --t3] * s2;
          return n3;
        }, c.prototype.readUint8 = c.prototype.readUInt8 = function(e3, t3) {
          return e3 >>>= 0, t3 || k(e3, 1, this.length), this[e3];
        }, c.prototype.readUint16LE = c.prototype.readUInt16LE = function(e3, t3) {
          return e3 >>>= 0, t3 || k(e3, 2, this.length), this[e3] | this[e3 + 1] << 8;
        }, c.prototype.readUint16BE = c.prototype.readUInt16BE = function(e3, t3) {
          return e3 >>>= 0, t3 || k(e3, 2, this.length), this[e3] << 8 | this[e3 + 1];
        }, c.prototype.readUint32LE = c.prototype.readUInt32LE = function(e3, t3) {
          return e3 >>>= 0, t3 || k(e3, 4, this.length), (this[e3] | this[e3 + 1] << 8 | this[e3 + 2] << 16) + 16777216 * this[e3 + 3];
        }, c.prototype.readUint32BE = c.prototype.readUInt32BE = function(e3, t3) {
          return e3 >>>= 0, t3 || k(e3, 4, this.length), 16777216 * this[e3] + (this[e3 + 1] << 16 | this[e3 + 2] << 8 | this[e3 + 3]);
        }, c.prototype.readBigUInt64LE = J(function(e3) {
          W(e3 >>>= 0, "offset");
          const t3 = this[e3], r3 = this[e3 + 7];
          void 0 !== t3 && void 0 !== r3 || K(e3, this.length - 8);
          const n3 = t3 + 256 * this[++e3] + 65536 * this[++e3] + this[++e3] * 2 ** 24, s2 = this[++e3] + 256 * this[++e3] + 65536 * this[++e3] + r3 * 2 ** 24;
          return BigInt(n3) + (BigInt(s2) << BigInt(32));
        }), c.prototype.readBigUInt64BE = J(function(e3) {
          W(e3 >>>= 0, "offset");
          const t3 = this[e3], r3 = this[e3 + 7];
          void 0 !== t3 && void 0 !== r3 || K(e3, this.length - 8);
          const n3 = t3 * 2 ** 24 + 65536 * this[++e3] + 256 * this[++e3] + this[++e3], s2 = this[++e3] * 2 ** 24 + 65536 * this[++e3] + 256 * this[++e3] + r3;
          return (BigInt(n3) << BigInt(32)) + BigInt(s2);
        }), c.prototype.readIntLE = function(e3, t3, r3) {
          e3 >>>= 0, t3 >>>= 0, r3 || k(e3, t3, this.length);
          let n3 = this[e3], s2 = 1, i2 = 0;
          for (; ++i2 < t3 && (s2 *= 256); )
            n3 += this[e3 + i2] * s2;
          return s2 *= 128, n3 >= s2 && (n3 -= Math.pow(2, 8 * t3)), n3;
        }, c.prototype.readIntBE = function(e3, t3, r3) {
          e3 >>>= 0, t3 >>>= 0, r3 || k(e3, t3, this.length);
          let n3 = t3, s2 = 1, i2 = this[e3 + --n3];
          for (; n3 > 0 && (s2 *= 256); )
            i2 += this[e3 + --n3] * s2;
          return s2 *= 128, i2 >= s2 && (i2 -= Math.pow(2, 8 * t3)), i2;
        }, c.prototype.readInt8 = function(e3, t3) {
          return e3 >>>= 0, t3 || k(e3, 1, this.length), 128 & this[e3] ? -1 * (255 - this[e3] + 1) : this[e3];
        }, c.prototype.readInt16LE = function(e3, t3) {
          e3 >>>= 0, t3 || k(e3, 2, this.length);
          const r3 = this[e3] | this[e3 + 1] << 8;
          return 32768 & r3 ? 4294901760 | r3 : r3;
        }, c.prototype.readInt16BE = function(e3, t3) {
          e3 >>>= 0, t3 || k(e3, 2, this.length);
          const r3 = this[e3 + 1] | this[e3] << 8;
          return 32768 & r3 ? 4294901760 | r3 : r3;
        }, c.prototype.readInt32LE = function(e3, t3) {
          return e3 >>>= 0, t3 || k(e3, 4, this.length), this[e3] | this[e3 + 1] << 8 | this[e3 + 2] << 16 | this[e3 + 3] << 24;
        }, c.prototype.readInt32BE = function(e3, t3) {
          return e3 >>>= 0, t3 || k(e3, 4, this.length), this[e3] << 24 | this[e3 + 1] << 16 | this[e3 + 2] << 8 | this[e3 + 3];
        }, c.prototype.readBigInt64LE = J(function(e3) {
          W(e3 >>>= 0, "offset");
          const t3 = this[e3], r3 = this[e3 + 7];
          void 0 !== t3 && void 0 !== r3 || K(e3, this.length - 8);
          const n3 = this[e3 + 4] + 256 * this[e3 + 5] + 65536 * this[e3 + 6] + (r3 << 24);
          return (BigInt(n3) << BigInt(32)) + BigInt(t3 + 256 * this[++e3] + 65536 * this[++e3] + this[++e3] * 2 ** 24);
        }), c.prototype.readBigInt64BE = J(function(e3) {
          W(e3 >>>= 0, "offset");
          const t3 = this[e3], r3 = this[e3 + 7];
          void 0 !== t3 && void 0 !== r3 || K(e3, this.length - 8);
          const n3 = (t3 << 24) + 65536 * this[++e3] + 256 * this[++e3] + this[++e3];
          return (BigInt(n3) << BigInt(32)) + BigInt(this[++e3] * 2 ** 24 + 65536 * this[++e3] + 256 * this[++e3] + r3);
        }), c.prototype.readFloatLE = function(e3, t3) {
          return e3 >>>= 0, t3 || k(e3, 4, this.length), s.read(this, e3, true, 23, 4);
        }, c.prototype.readFloatBE = function(e3, t3) {
          return e3 >>>= 0, t3 || k(e3, 4, this.length), s.read(this, e3, false, 23, 4);
        }, c.prototype.readDoubleLE = function(e3, t3) {
          return e3 >>>= 0, t3 || k(e3, 8, this.length), s.read(this, e3, true, 52, 8);
        }, c.prototype.readDoubleBE = function(e3, t3) {
          return e3 >>>= 0, t3 || k(e3, 8, this.length), s.read(this, e3, false, 52, 8);
        }, c.prototype.writeUintLE = c.prototype.writeUIntLE = function(e3, t3, r3, n3) {
          e3 = +e3, t3 >>>= 0, r3 >>>= 0, n3 || L(this, e3, t3, r3, Math.pow(2, 8 * r3) - 1, 0);
          let s2 = 1, i2 = 0;
          for (this[t3] = 255 & e3; ++i2 < r3 && (s2 *= 256); )
            this[t3 + i2] = e3 / s2 & 255;
          return t3 + r3;
        }, c.prototype.writeUintBE = c.prototype.writeUIntBE = function(e3, t3, r3, n3) {
          e3 = +e3, t3 >>>= 0, r3 >>>= 0, n3 || L(this, e3, t3, r3, Math.pow(2, 8 * r3) - 1, 0);
          let s2 = r3 - 1, i2 = 1;
          for (this[t3 + s2] = 255 & e3; --s2 >= 0 && (i2 *= 256); )
            this[t3 + s2] = e3 / i2 & 255;
          return t3 + r3;
        }, c.prototype.writeUint8 = c.prototype.writeUInt8 = function(e3, t3, r3) {
          return e3 = +e3, t3 >>>= 0, r3 || L(this, e3, t3, 1, 255, 0), this[t3] = 255 & e3, t3 + 1;
        }, c.prototype.writeUint16LE = c.prototype.writeUInt16LE = function(e3, t3, r3) {
          return e3 = +e3, t3 >>>= 0, r3 || L(this, e3, t3, 2, 65535, 0), this[t3] = 255 & e3, this[t3 + 1] = e3 >>> 8, t3 + 2;
        }, c.prototype.writeUint16BE = c.prototype.writeUInt16BE = function(e3, t3, r3) {
          return e3 = +e3, t3 >>>= 0, r3 || L(this, e3, t3, 2, 65535, 0), this[t3] = e3 >>> 8, this[t3 + 1] = 255 & e3, t3 + 2;
        }, c.prototype.writeUint32LE = c.prototype.writeUInt32LE = function(e3, t3, r3) {
          return e3 = +e3, t3 >>>= 0, r3 || L(this, e3, t3, 4, 4294967295, 0), this[t3 + 3] = e3 >>> 24, this[t3 + 2] = e3 >>> 16, this[t3 + 1] = e3 >>> 8, this[t3] = 255 & e3, t3 + 4;
        }, c.prototype.writeUint32BE = c.prototype.writeUInt32BE = function(e3, t3, r3) {
          return e3 = +e3, t3 >>>= 0, r3 || L(this, e3, t3, 4, 4294967295, 0), this[t3] = e3 >>> 24, this[t3 + 1] = e3 >>> 16, this[t3 + 2] = e3 >>> 8, this[t3 + 3] = 255 & e3, t3 + 4;
        }, c.prototype.writeBigUInt64LE = J(function(e3, t3 = 0) {
          return D(this, e3, t3, BigInt(0), BigInt("0xffffffffffffffff"));
        }), c.prototype.writeBigUInt64BE = J(function(e3, t3 = 0) {
          return P(this, e3, t3, BigInt(0), BigInt("0xffffffffffffffff"));
        }), c.prototype.writeIntLE = function(e3, t3, r3, n3) {
          if (e3 = +e3, t3 >>>= 0, !n3) {
            const n4 = Math.pow(2, 8 * r3 - 1);
            L(this, e3, t3, r3, n4 - 1, -n4);
          }
          let s2 = 0, i2 = 1, o2 = 0;
          for (this[t3] = 255 & e3; ++s2 < r3 && (i2 *= 256); )
            e3 < 0 && 0 === o2 && 0 !== this[t3 + s2 - 1] && (o2 = 1), this[t3 + s2] = (e3 / i2 >> 0) - o2 & 255;
          return t3 + r3;
        }, c.prototype.writeIntBE = function(e3, t3, r3, n3) {
          if (e3 = +e3, t3 >>>= 0, !n3) {
            const n4 = Math.pow(2, 8 * r3 - 1);
            L(this, e3, t3, r3, n4 - 1, -n4);
          }
          let s2 = r3 - 1, i2 = 1, o2 = 0;
          for (this[t3 + s2] = 255 & e3; --s2 >= 0 && (i2 *= 256); )
            e3 < 0 && 0 === o2 && 0 !== this[t3 + s2 + 1] && (o2 = 1), this[t3 + s2] = (e3 / i2 >> 0) - o2 & 255;
          return t3 + r3;
        }, c.prototype.writeInt8 = function(e3, t3, r3) {
          return e3 = +e3, t3 >>>= 0, r3 || L(this, e3, t3, 1, 127, -128), e3 < 0 && (e3 = 255 + e3 + 1), this[t3] = 255 & e3, t3 + 1;
        }, c.prototype.writeInt16LE = function(e3, t3, r3) {
          return e3 = +e3, t3 >>>= 0, r3 || L(this, e3, t3, 2, 32767, -32768), this[t3] = 255 & e3, this[t3 + 1] = e3 >>> 8, t3 + 2;
        }, c.prototype.writeInt16BE = function(e3, t3, r3) {
          return e3 = +e3, t3 >>>= 0, r3 || L(this, e3, t3, 2, 32767, -32768), this[t3] = e3 >>> 8, this[t3 + 1] = 255 & e3, t3 + 2;
        }, c.prototype.writeInt32LE = function(e3, t3, r3) {
          return e3 = +e3, t3 >>>= 0, r3 || L(this, e3, t3, 4, 2147483647, -2147483648), this[t3] = 255 & e3, this[t3 + 1] = e3 >>> 8, this[t3 + 2] = e3 >>> 16, this[t3 + 3] = e3 >>> 24, t3 + 4;
        }, c.prototype.writeInt32BE = function(e3, t3, r3) {
          return e3 = +e3, t3 >>>= 0, r3 || L(this, e3, t3, 4, 2147483647, -2147483648), e3 < 0 && (e3 = 4294967295 + e3 + 1), this[t3] = e3 >>> 24, this[t3 + 1] = e3 >>> 16, this[t3 + 2] = e3 >>> 8, this[t3 + 3] = 255 & e3, t3 + 4;
        }, c.prototype.writeBigInt64LE = J(function(e3, t3 = 0) {
          return D(this, e3, t3, -BigInt("0x8000000000000000"), BigInt("0x7fffffffffffffff"));
        }), c.prototype.writeBigInt64BE = J(function(e3, t3 = 0) {
          return P(this, e3, t3, -BigInt("0x8000000000000000"), BigInt("0x7fffffffffffffff"));
        }), c.prototype.writeFloatLE = function(e3, t3, r3) {
          return B(this, e3, t3, true, r3);
        }, c.prototype.writeFloatBE = function(e3, t3, r3) {
          return B(this, e3, t3, false, r3);
        }, c.prototype.writeDoubleLE = function(e3, t3, r3) {
          return M(this, e3, t3, true, r3);
        }, c.prototype.writeDoubleBE = function(e3, t3, r3) {
          return M(this, e3, t3, false, r3);
        }, c.prototype.copy = function(e3, t3, r3, n3) {
          if (!c.isBuffer(e3))
            throw new TypeError("argument should be a Buffer");
          if (r3 || (r3 = 0), n3 || 0 === n3 || (n3 = this.length), t3 >= e3.length && (t3 = e3.length), t3 || (t3 = 0), n3 > 0 && n3 < r3 && (n3 = r3), n3 === r3)
            return 0;
          if (0 === e3.length || 0 === this.length)
            return 0;
          if (t3 < 0)
            throw new RangeError("targetStart out of bounds");
          if (r3 < 0 || r3 >= this.length)
            throw new RangeError("Index out of range");
          if (n3 < 0)
            throw new RangeError("sourceEnd out of bounds");
          n3 > this.length && (n3 = this.length), e3.length - t3 < n3 - r3 && (n3 = e3.length - t3 + r3);
          const s2 = n3 - r3;
          return this === e3 && "function" == typeof Uint8Array.prototype.copyWithin ? this.copyWithin(t3, r3, n3) : Uint8Array.prototype.set.call(e3, this.subarray(r3, n3), t3), s2;
        }, c.prototype.fill = function(e3, t3, r3, n3) {
          if ("string" == typeof e3) {
            if ("string" == typeof t3 ? (n3 = t3, t3 = 0, r3 = this.length) : "string" == typeof r3 && (n3 = r3, r3 = this.length), void 0 !== n3 && "string" != typeof n3)
              throw new TypeError("encoding must be a string");
            if ("string" == typeof n3 && !c.isEncoding(n3))
              throw new TypeError("Unknown encoding: " + n3);
            if (1 === e3.length) {
              const t4 = e3.charCodeAt(0);
              ("utf8" === n3 && t4 < 128 || "latin1" === n3) && (e3 = t4);
            }
          } else
            "number" == typeof e3 ? e3 &= 255 : "boolean" == typeof e3 && (e3 = Number(e3));
          if (t3 < 0 || this.length < t3 || this.length < r3)
            throw new RangeError("Out of range index");
          if (r3 <= t3)
            return this;
          let s2;
          if (t3 >>>= 0, r3 = void 0 === r3 ? this.length : r3 >>> 0, e3 || (e3 = 0), "number" == typeof e3)
            for (s2 = t3; s2 < r3; ++s2)
              this[s2] = e3;
          else {
            const i2 = c.isBuffer(e3) ? e3 : c.from(e3, n3), o2 = i2.length;
            if (0 === o2)
              throw new TypeError('The value "' + e3 + '" is invalid for argument "value"');
            for (s2 = 0; s2 < r3 - t3; ++s2)
              this[s2 + t3] = i2[s2 % o2];
          }
          return this;
        };
        const U = {};
        function H(e3, t3, r3) {
          U[e3] = class extends r3 {
            constructor() {
              super(), Object.defineProperty(this, "message", { value: t3.apply(this, arguments), writable: true, configurable: true }), this.name = `${this.name} [${e3}]`, this.stack, delete this.name;
            }
            get code() {
              return e3;
            }
            set code(e4) {
              Object.defineProperty(this, "code", { configurable: true, enumerable: true, value: e4, writable: true });
            }
            toString() {
              return `${this.name} [${e3}]: ${this.message}`;
            }
          };
        }
        function z(e3) {
          let t3 = "", r3 = e3.length;
          const n3 = "-" === e3[0] ? 1 : 0;
          for (; r3 >= n3 + 4; r3 -= 3)
            t3 = `_${e3.slice(r3 - 3, r3)}${t3}`;
          return `${e3.slice(0, r3)}${t3}`;
        }
        function j(e3, t3, r3, n3, s2, i2) {
          if (e3 > r3 || e3 < t3) {
            const n4 = "bigint" == typeof t3 ? "n" : "";
            let s3;
            throw s3 = i2 > 3 ? 0 === t3 || t3 === BigInt(0) ? `>= 0${n4} and < 2${n4} ** ${8 * (i2 + 1)}${n4}` : `>= -(2${n4} ** ${8 * (i2 + 1) - 1}${n4}) and < 2 ** ${8 * (i2 + 1) - 1}${n4}` : `>= ${t3}${n4} and <= ${r3}${n4}`, new U.ERR_OUT_OF_RANGE("value", s3, e3);
          }
          !function(e4, t4, r4) {
            W(t4, "offset"), void 0 !== e4[t4] && void 0 !== e4[t4 + r4] || K(t4, e4.length - (r4 + 1));
          }(n3, s2, i2);
        }
        function W(e3, t3) {
          if ("number" != typeof e3)
            throw new U.ERR_INVALID_ARG_TYPE(t3, "number", e3);
        }
        function K(e3, t3, r3) {
          if (Math.floor(e3) !== e3)
            throw W(e3, r3), new U.ERR_OUT_OF_RANGE(r3 || "offset", "an integer", e3);
          if (t3 < 0)
            throw new U.ERR_BUFFER_OUT_OF_BOUNDS();
          throw new U.ERR_OUT_OF_RANGE(r3 || "offset", `>= ${r3 ? 1 : 0} and <= ${t3}`, e3);
        }
        H("ERR_BUFFER_OUT_OF_BOUNDS", function(e3) {
          return e3 ? `${e3} is outside of buffer bounds` : "Attempt to access memory outside buffer bounds";
        }, RangeError), H("ERR_INVALID_ARG_TYPE", function(e3, t3) {
          return `The "${e3}" argument must be of type number. Received type ${typeof t3}`;
        }, TypeError), H("ERR_OUT_OF_RANGE", function(e3, t3, r3) {
          let n3 = `The value of "${e3}" is out of range.`, s2 = r3;
          return Number.isInteger(r3) && Math.abs(r3) > 2 ** 32 ? s2 = z(String(r3)) : "bigint" == typeof r3 && (s2 = String(r3), (r3 > BigInt(2) ** BigInt(32) || r3 < -(BigInt(2) ** BigInt(32))) && (s2 = z(s2)), s2 += "n"), n3 += ` It must be ${t3}. Received ${s2}`, n3;
        }, RangeError);
        const G = /[^+/0-9A-Za-z-_]/g;
        function V(e3, t3) {
          let r3;
          t3 = t3 || 1 / 0;
          const n3 = e3.length;
          let s2 = null;
          const i2 = [];
          for (let o2 = 0; o2 < n3; ++o2) {
            if (r3 = e3.charCodeAt(o2), r3 > 55295 && r3 < 57344) {
              if (!s2) {
                if (r3 > 56319) {
                  (t3 -= 3) > -1 && i2.push(239, 191, 189);
                  continue;
                }
                if (o2 + 1 === n3) {
                  (t3 -= 3) > -1 && i2.push(239, 191, 189);
                  continue;
                }
                s2 = r3;
                continue;
              }
              if (r3 < 56320) {
                (t3 -= 3) > -1 && i2.push(239, 191, 189), s2 = r3;
                continue;
              }
              r3 = 65536 + (s2 - 55296 << 10 | r3 - 56320);
            } else
              s2 && (t3 -= 3) > -1 && i2.push(239, 191, 189);
            if (s2 = null, r3 < 128) {
              if ((t3 -= 1) < 0)
                break;
              i2.push(r3);
            } else if (r3 < 2048) {
              if ((t3 -= 2) < 0)
                break;
              i2.push(r3 >> 6 | 192, 63 & r3 | 128);
            } else if (r3 < 65536) {
              if ((t3 -= 3) < 0)
                break;
              i2.push(r3 >> 12 | 224, r3 >> 6 & 63 | 128, 63 & r3 | 128);
            } else {
              if (!(r3 < 1114112))
                throw new Error("Invalid code point");
              if ((t3 -= 4) < 0)
                break;
              i2.push(r3 >> 18 | 240, r3 >> 12 & 63 | 128, r3 >> 6 & 63 | 128, 63 & r3 | 128);
            }
          }
          return i2;
        }
        function $(e3) {
          return n2.toByteArray(function(e4) {
            if ((e4 = (e4 = e4.split("=")[0]).trim().replace(G, "")).length < 2)
              return "";
            for (; e4.length % 4 != 0; )
              e4 += "=";
            return e4;
          }(e3));
        }
        function X(e3, t3, r3, n3) {
          let s2;
          for (s2 = 0; s2 < n3 && !(s2 + r3 >= t3.length || s2 >= e3.length); ++s2)
            t3[s2 + r3] = e3[s2];
          return s2;
        }
        function q(e3, t3) {
          return e3 instanceof t3 || null != e3 && null != e3.constructor && null != e3.constructor.name && e3.constructor.name === t3.name;
        }
        function Z(e3) {
          return e3 != e3;
        }
        const Y = function() {
          const e3 = "0123456789abcdef", t3 = new Array(256);
          for (let r3 = 0; r3 < 16; ++r3) {
            const n3 = 16 * r3;
            for (let s2 = 0; s2 < 16; ++s2)
              t3[n3 + s2] = e3[r3] + e3[s2];
          }
          return t3;
        }();
        function J(e3) {
          return "undefined" == typeof BigInt ? Q : e3;
        }
        function Q() {
          throw new Error("BigInt not supported");
        }
      }, 7187: (e2) => {
        "use strict";
        var t2, r2 = "object" == typeof Reflect ? Reflect : null, n2 = r2 && "function" == typeof r2.apply ? r2.apply : function(e3, t3, r3) {
          return Function.prototype.apply.call(e3, t3, r3);
        };
        t2 = r2 && "function" == typeof r2.ownKeys ? r2.ownKeys : Object.getOwnPropertySymbols ? function(e3) {
          return Object.getOwnPropertyNames(e3).concat(Object.getOwnPropertySymbols(e3));
        } : function(e3) {
          return Object.getOwnPropertyNames(e3);
        };
        var s = Number.isNaN || function(e3) {
          return e3 != e3;
        };
        function i() {
          i.init.call(this);
        }
        e2.exports = i, e2.exports.once = function(e3, t3) {
          return new Promise(function(r3, n3) {
            function s2(r4) {
              e3.removeListener(t3, i2), n3(r4);
            }
            function i2() {
              "function" == typeof e3.removeListener && e3.removeListener("error", s2), r3([].slice.call(arguments));
            }
            m(e3, t3, i2, { once: true }), "error" !== t3 && function(e4, t4, r4) {
              "function" == typeof e4.on && m(e4, "error", t4, { once: true });
            }(e3, s2);
          });
        }, i.EventEmitter = i, i.prototype._events = void 0, i.prototype._eventsCount = 0, i.prototype._maxListeners = void 0;
        var o = 10;
        function a(e3) {
          if ("function" != typeof e3)
            throw new TypeError('The "listener" argument must be of type Function. Received type ' + typeof e3);
        }
        function c(e3) {
          return void 0 === e3._maxListeners ? i.defaultMaxListeners : e3._maxListeners;
        }
        function u(e3, t3, r3, n3) {
          var s2, i2, o2, u2;
          if (a(r3), void 0 === (i2 = e3._events) ? (i2 = e3._events = /* @__PURE__ */ Object.create(null), e3._eventsCount = 0) : (void 0 !== i2.newListener && (e3.emit("newListener", t3, r3.listener ? r3.listener : r3), i2 = e3._events), o2 = i2[t3]), void 0 === o2)
            o2 = i2[t3] = r3, ++e3._eventsCount;
          else if ("function" == typeof o2 ? o2 = i2[t3] = n3 ? [r3, o2] : [o2, r3] : n3 ? o2.unshift(r3) : o2.push(r3), (s2 = c(e3)) > 0 && o2.length > s2 && !o2.warned) {
            o2.warned = true;
            var l2 = new Error("Possible EventEmitter memory leak detected. " + o2.length + " " + String(t3) + " listeners added. Use emitter.setMaxListeners() to increase limit");
            l2.name = "MaxListenersExceededWarning", l2.emitter = e3, l2.type = t3, l2.count = o2.length, u2 = l2, console && console.warn && console.warn(u2);
          }
          return e3;
        }
        function l() {
          if (!this.fired)
            return this.target.removeListener(this.type, this.wrapFn), this.fired = true, 0 === arguments.length ? this.listener.call(this.target) : this.listener.apply(this.target, arguments);
        }
        function h(e3, t3, r3) {
          var n3 = { fired: false, wrapFn: void 0, target: e3, type: t3, listener: r3 }, s2 = l.bind(n3);
          return s2.listener = r3, n3.wrapFn = s2, s2;
        }
        function p(e3, t3, r3) {
          var n3 = e3._events;
          if (void 0 === n3)
            return [];
          var s2 = n3[t3];
          return void 0 === s2 ? [] : "function" == typeof s2 ? r3 ? [s2.listener || s2] : [s2] : r3 ? function(e4) {
            for (var t4 = new Array(e4.length), r4 = 0; r4 < t4.length; ++r4)
              t4[r4] = e4[r4].listener || e4[r4];
            return t4;
          }(s2) : f(s2, s2.length);
        }
        function d(e3) {
          var t3 = this._events;
          if (void 0 !== t3) {
            var r3 = t3[e3];
            if ("function" == typeof r3)
              return 1;
            if (void 0 !== r3)
              return r3.length;
          }
          return 0;
        }
        function f(e3, t3) {
          for (var r3 = new Array(t3), n3 = 0; n3 < t3; ++n3)
            r3[n3] = e3[n3];
          return r3;
        }
        function m(e3, t3, r3, n3) {
          if ("function" == typeof e3.on)
            n3.once ? e3.once(t3, r3) : e3.on(t3, r3);
          else {
            if ("function" != typeof e3.addEventListener)
              throw new TypeError('The "emitter" argument must be of type EventEmitter. Received type ' + typeof e3);
            e3.addEventListener(t3, function s2(i2) {
              n3.once && e3.removeEventListener(t3, s2), r3(i2);
            });
          }
        }
        Object.defineProperty(i, "defaultMaxListeners", { enumerable: true, get: function() {
          return o;
        }, set: function(e3) {
          if ("number" != typeof e3 || e3 < 0 || s(e3))
            throw new RangeError('The value of "defaultMaxListeners" is out of range. It must be a non-negative number. Received ' + e3 + ".");
          o = e3;
        } }), i.init = function() {
          void 0 !== this._events && this._events !== Object.getPrototypeOf(this)._events || (this._events = /* @__PURE__ */ Object.create(null), this._eventsCount = 0), this._maxListeners = this._maxListeners || void 0;
        }, i.prototype.setMaxListeners = function(e3) {
          if ("number" != typeof e3 || e3 < 0 || s(e3))
            throw new RangeError('The value of "n" is out of range. It must be a non-negative number. Received ' + e3 + ".");
          return this._maxListeners = e3, this;
        }, i.prototype.getMaxListeners = function() {
          return c(this);
        }, i.prototype.emit = function(e3) {
          for (var t3 = [], r3 = 1; r3 < arguments.length; r3++)
            t3.push(arguments[r3]);
          var s2 = "error" === e3, i2 = this._events;
          if (void 0 !== i2)
            s2 = s2 && void 0 === i2.error;
          else if (!s2)
            return false;
          if (s2) {
            var o2;
            if (t3.length > 0 && (o2 = t3[0]), o2 instanceof Error)
              throw o2;
            var a2 = new Error("Unhandled error." + (o2 ? " (" + o2.message + ")" : ""));
            throw a2.context = o2, a2;
          }
          var c2 = i2[e3];
          if (void 0 === c2)
            return false;
          if ("function" == typeof c2)
            n2(c2, this, t3);
          else {
            var u2 = c2.length, l2 = f(c2, u2);
            for (r3 = 0; r3 < u2; ++r3)
              n2(l2[r3], this, t3);
          }
          return true;
        }, i.prototype.addListener = function(e3, t3) {
          return u(this, e3, t3, false);
        }, i.prototype.on = i.prototype.addListener, i.prototype.prependListener = function(e3, t3) {
          return u(this, e3, t3, true);
        }, i.prototype.once = function(e3, t3) {
          return a(t3), this.on(e3, h(this, e3, t3)), this;
        }, i.prototype.prependOnceListener = function(e3, t3) {
          return a(t3), this.prependListener(e3, h(this, e3, t3)), this;
        }, i.prototype.removeListener = function(e3, t3) {
          var r3, n3, s2, i2, o2;
          if (a(t3), void 0 === (n3 = this._events))
            return this;
          if (void 0 === (r3 = n3[e3]))
            return this;
          if (r3 === t3 || r3.listener === t3)
            0 == --this._eventsCount ? this._events = /* @__PURE__ */ Object.create(null) : (delete n3[e3], n3.removeListener && this.emit("removeListener", e3, r3.listener || t3));
          else if ("function" != typeof r3) {
            for (s2 = -1, i2 = r3.length - 1; i2 >= 0; i2--)
              if (r3[i2] === t3 || r3[i2].listener === t3) {
                o2 = r3[i2].listener, s2 = i2;
                break;
              }
            if (s2 < 0)
              return this;
            0 === s2 ? r3.shift() : function(e4, t4) {
              for (; t4 + 1 < e4.length; t4++)
                e4[t4] = e4[t4 + 1];
              e4.pop();
            }(r3, s2), 1 === r3.length && (n3[e3] = r3[0]), void 0 !== n3.removeListener && this.emit("removeListener", e3, o2 || t3);
          }
          return this;
        }, i.prototype.off = i.prototype.removeListener, i.prototype.removeAllListeners = function(e3) {
          var t3, r3, n3;
          if (void 0 === (r3 = this._events))
            return this;
          if (void 0 === r3.removeListener)
            return 0 === arguments.length ? (this._events = /* @__PURE__ */ Object.create(null), this._eventsCount = 0) : void 0 !== r3[e3] && (0 == --this._eventsCount ? this._events = /* @__PURE__ */ Object.create(null) : delete r3[e3]), this;
          if (0 === arguments.length) {
            var s2, i2 = Object.keys(r3);
            for (n3 = 0; n3 < i2.length; ++n3)
              "removeListener" !== (s2 = i2[n3]) && this.removeAllListeners(s2);
            return this.removeAllListeners("removeListener"), this._events = /* @__PURE__ */ Object.create(null), this._eventsCount = 0, this;
          }
          if ("function" == typeof (t3 = r3[e3]))
            this.removeListener(e3, t3);
          else if (void 0 !== t3)
            for (n3 = t3.length - 1; n3 >= 0; n3--)
              this.removeListener(e3, t3[n3]);
          return this;
        }, i.prototype.listeners = function(e3) {
          return p(this, e3, true);
        }, i.prototype.rawListeners = function(e3) {
          return p(this, e3, false);
        }, i.listenerCount = function(e3, t3) {
          return "function" == typeof e3.listenerCount ? e3.listenerCount(t3) : d.call(e3, t3);
        }, i.prototype.listenerCount = d, i.prototype.eventNames = function() {
          return this._eventsCount > 0 ? t2(this._events) : [];
        };
      }, 645: (e2, t2) => {
        t2.read = function(e3, t3, r2, n2, s) {
          var i, o, a = 8 * s - n2 - 1, c = (1 << a) - 1, u = c >> 1, l = -7, h = r2 ? s - 1 : 0, p = r2 ? -1 : 1, d = e3[t3 + h];
          for (h += p, i = d & (1 << -l) - 1, d >>= -l, l += a; l > 0; i = 256 * i + e3[t3 + h], h += p, l -= 8)
            ;
          for (o = i & (1 << -l) - 1, i >>= -l, l += n2; l > 0; o = 256 * o + e3[t3 + h], h += p, l -= 8)
            ;
          if (0 === i)
            i = 1 - u;
          else {
            if (i === c)
              return o ? NaN : 1 / 0 * (d ? -1 : 1);
            o += Math.pow(2, n2), i -= u;
          }
          return (d ? -1 : 1) * o * Math.pow(2, i - n2);
        }, t2.write = function(e3, t3, r2, n2, s, i) {
          var o, a, c, u = 8 * i - s - 1, l = (1 << u) - 1, h = l >> 1, p = 23 === s ? Math.pow(2, -24) - Math.pow(2, -77) : 0, d = n2 ? 0 : i - 1, f = n2 ? 1 : -1, m = t3 < 0 || 0 === t3 && 1 / t3 < 0 ? 1 : 0;
          for (t3 = Math.abs(t3), isNaN(t3) || t3 === 1 / 0 ? (a = isNaN(t3) ? 1 : 0, o = l) : (o = Math.floor(Math.log(t3) / Math.LN2), t3 * (c = Math.pow(2, -o)) < 1 && (o--, c *= 2), (t3 += o + h >= 1 ? p / c : p * Math.pow(2, 1 - h)) * c >= 2 && (o++, c /= 2), o + h >= l ? (a = 0, o = l) : o + h >= 1 ? (a = (t3 * c - 1) * Math.pow(2, s), o += h) : (a = t3 * Math.pow(2, h - 1) * Math.pow(2, s), o = 0)); s >= 8; e3[r2 + d] = 255 & a, d += f, a /= 256, s -= 8)
            ;
          for (o = o << s | a, u += s; u > 0; e3[r2 + d] = 255 & o, d += f, o /= 256, u -= 8)
            ;
          e3[r2 + d - f] |= 128 * m;
        };
      }, 5705: (e2, t2, r2) => {
        "use strict";
        var n2, s, i = r2.g.MutationObserver || r2.g.WebKitMutationObserver;
        if (i) {
          var o = 0, a = new i(h), c = r2.g.document.createTextNode("");
          a.observe(c, { characterData: true }), n2 = function() {
            c.data = o = ++o % 2;
          };
        } else if (r2.g.setImmediate || void 0 === r2.g.MessageChannel)
          n2 = "document" in r2.g && "onreadystatechange" in r2.g.document.createElement("script") ? function() {
            var e3 = r2.g.document.createElement("script");
            e3.onreadystatechange = function() {
              h(), e3.onreadystatechange = null, e3.parentNode.removeChild(e3), e3 = null;
            }, r2.g.document.documentElement.appendChild(e3);
          } : function() {
            setTimeout(h, 0);
          };
        else {
          var u = new r2.g.MessageChannel();
          u.port1.onmessage = h, n2 = function() {
            u.port2.postMessage(0);
          };
        }
        var l = [];
        function h() {
          var e3, t3;
          s = true;
          for (var r3 = l.length; r3; ) {
            for (t3 = l, l = [], e3 = -1; ++e3 < r3; )
              t3[e3]();
            r3 = l.length;
          }
          s = false;
        }
        e2.exports = function(e3) {
          1 !== l.push(e3) || s || n2();
        };
      }, 5717: (e2) => {
        "function" == typeof Object.create ? e2.exports = function(e3, t2) {
          t2 && (e3.super_ = t2, e3.prototype = Object.create(t2.prototype, { constructor: { value: e3, enumerable: false, writable: true, configurable: true } }));
        } : e2.exports = function(e3, t2) {
          if (t2) {
            e3.super_ = t2;
            var r2 = function() {
            };
            r2.prototype = t2.prototype, e3.prototype = new r2(), e3.prototype.constructor = e3;
          }
        };
      }, 8458: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(8910), s = r2(3790), i = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=";
        t2.encode = function(e3) {
          for (var t3, r3, s2, o, a, c, u, l = [], h = 0, p = e3.length, d = p, f = "string" !== n2.getTypeOf(e3); h < e3.length; )
            d = p - h, f ? (t3 = e3[h++], r3 = h < p ? e3[h++] : 0, s2 = h < p ? e3[h++] : 0) : (t3 = e3.charCodeAt(h++), r3 = h < p ? e3.charCodeAt(h++) : 0, s2 = h < p ? e3.charCodeAt(h++) : 0), o = t3 >> 2, a = (3 & t3) << 4 | r3 >> 4, c = d > 1 ? (15 & r3) << 2 | s2 >> 6 : 64, u = d > 2 ? 63 & s2 : 64, l.push(i.charAt(o) + i.charAt(a) + i.charAt(c) + i.charAt(u));
          return l.join("");
        }, t2.decode = function(e3) {
          var t3, r3, n3, o, a, c, u = 0, l = 0, h = "data:";
          if (e3.substr(0, h.length) === h)
            throw new Error("Invalid base64 input, it looks like a data url.");
          var p, d = 3 * (e3 = e3.replace(/[^A-Za-z0-9+/=]/g, "")).length / 4;
          if (e3.charAt(e3.length - 1) === i.charAt(64) && d--, e3.charAt(e3.length - 2) === i.charAt(64) && d--, d % 1 != 0)
            throw new Error("Invalid base64 input, bad content length.");
          for (p = s.uint8array ? new Uint8Array(0 | d) : new Array(0 | d); u < e3.length; )
            t3 = i.indexOf(e3.charAt(u++)) << 2 | (o = i.indexOf(e3.charAt(u++))) >> 4, r3 = (15 & o) << 4 | (a = i.indexOf(e3.charAt(u++))) >> 2, n3 = (3 & a) << 6 | (c = i.indexOf(e3.charAt(u++))), p[l++] = t3, 64 !== a && (p[l++] = r3), 64 !== c && (p[l++] = n3);
          return p;
        };
      }, 7326: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(8565), s = r2(5301), i = r2(2541), o = r2(5977);
        function a(e3, t3, r3, n3, s2) {
          this.compressedSize = e3, this.uncompressedSize = t3, this.crc32 = r3, this.compression = n3, this.compressedContent = s2;
        }
        a.prototype = { getContentWorker: function() {
          var e3 = new s(n2.Promise.resolve(this.compressedContent)).pipe(this.compression.uncompressWorker()).pipe(new o("data_length")), t3 = this;
          return e3.on("end", function() {
            if (this.streamInfo.data_length !== t3.uncompressedSize)
              throw new Error("Bug : uncompressed data size mismatch");
          }), e3;
        }, getCompressedWorker: function() {
          return new s(n2.Promise.resolve(this.compressedContent)).withStreamInfo("compressedSize", this.compressedSize).withStreamInfo("uncompressedSize", this.uncompressedSize).withStreamInfo("crc32", this.crc32).withStreamInfo("compression", this.compression);
        } }, a.createWorkerFrom = function(e3, t3, r3) {
          return e3.pipe(new i()).pipe(new o("uncompressedSize")).pipe(t3.compressWorker(r3)).pipe(new o("compressedSize")).withStreamInfo("compression", t3);
        }, e2.exports = a;
      }, 1678: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(3718);
        t2.STORE = { magic: "\0\0", compressWorker: function() {
          return new n2("STORE compression");
        }, uncompressWorker: function() {
          return new n2("STORE decompression");
        } }, t2.DEFLATE = r2(1033);
      }, 6988: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(8910), s = function() {
          for (var e3, t3 = [], r3 = 0; r3 < 256; r3++) {
            e3 = r3;
            for (var n3 = 0; n3 < 8; n3++)
              e3 = 1 & e3 ? 3988292384 ^ e3 >>> 1 : e3 >>> 1;
            t3[r3] = e3;
          }
          return t3;
        }();
        e2.exports = function(e3, t3) {
          return void 0 !== e3 && e3.length ? "string" !== n2.getTypeOf(e3) ? function(e4, t4, r3, n3) {
            var i = s, o = 0 + r3;
            e4 ^= -1;
            for (var a = 0; a < o; a++)
              e4 = e4 >>> 8 ^ i[255 & (e4 ^ t4[a])];
            return -1 ^ e4;
          }(0 | t3, e3, e3.length) : function(e4, t4, r3, n3) {
            var i = s, o = 0 + r3;
            e4 ^= -1;
            for (var a = 0; a < o; a++)
              e4 = e4 >>> 8 ^ i[255 & (e4 ^ t4.charCodeAt(a))];
            return -1 ^ e4;
          }(0 | t3, e3, e3.length) : 0;
        };
      }, 6032: (e2, t2) => {
        "use strict";
        t2.base64 = false, t2.binary = false, t2.dir = false, t2.createFolders = true, t2.date = null, t2.compression = null, t2.compressionOptions = null, t2.comment = null, t2.unixPermissions = null, t2.dosPermissions = null;
      }, 8565: (e2, t2, r2) => {
        "use strict";
        var n2;
        n2 = "undefined" != typeof Promise ? Promise : r2(3389), e2.exports = { Promise: n2 };
      }, 1033: (e2, t2, r2) => {
        "use strict";
        var n2 = "undefined" != typeof Uint8Array && "undefined" != typeof Uint16Array && "undefined" != typeof Uint32Array, s = r2(9591), i = r2(8910), o = r2(3718), a = n2 ? "uint8array" : "array";
        function c(e3, t3) {
          o.call(this, "FlateWorker/" + e3), this._pako = null, this._pakoAction = e3, this._pakoOptions = t3, this.meta = {};
        }
        t2.magic = "\b\0", i.inherits(c, o), c.prototype.processChunk = function(e3) {
          this.meta = e3.meta, null === this._pako && this._createPako(), this._pako.push(i.transformTo(a, e3.data), false);
        }, c.prototype.flush = function() {
          o.prototype.flush.call(this), null === this._pako && this._createPako(), this._pako.push([], true);
        }, c.prototype.cleanUp = function() {
          o.prototype.cleanUp.call(this), this._pako = null;
        }, c.prototype._createPako = function() {
          this._pako = new s[this._pakoAction]({ raw: true, level: this._pakoOptions.level || -1 });
          var e3 = this;
          this._pako.onData = function(t3) {
            e3.push({ data: t3, meta: e3.meta });
          };
        }, t2.compressWorker = function(e3) {
          return new c("Deflate", e3);
        }, t2.uncompressWorker = function() {
          return new c("Inflate", {});
        };
      }, 4979: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(8910), s = r2(3718), i = r2(3600), o = r2(6988), a = r2(1141), c = function(e3, t3) {
          var r3, n3 = "";
          for (r3 = 0; r3 < t3; r3++)
            n3 += String.fromCharCode(255 & e3), e3 >>>= 8;
          return n3;
        }, u = function(e3, t3, r3, s2, u2, l2) {
          var h2, p, d = e3.file, f = e3.compression, m = l2 !== i.utf8encode, w = n2.transformTo("string", l2(d.name)), g = n2.transformTo("string", i.utf8encode(d.name)), y = d.comment, b = n2.transformTo("string", l2(y)), x = n2.transformTo("string", i.utf8encode(y)), v = g.length !== d.name.length, _ = x.length !== y.length, E = "", T = "", A = "", S = d.dir, I = d.date, R = { crc32: 0, compressedSize: 0, uncompressedSize: 0 };
          t3 && !r3 || (R.crc32 = e3.crc32, R.compressedSize = e3.compressedSize, R.uncompressedSize = e3.uncompressedSize);
          var N = 0;
          t3 && (N |= 8), m || !v && !_ || (N |= 2048);
          var C, O, k = 0, L = 0;
          S && (k |= 16), "UNIX" === u2 ? (L = 798, k |= (O = C = d.unixPermissions, C || (O = S ? 16893 : 33204), (65535 & O) << 16)) : (L = 20, k |= 63 & (d.dosPermissions || 0)), h2 = I.getUTCHours(), h2 <<= 6, h2 |= I.getUTCMinutes(), h2 <<= 5, h2 |= I.getUTCSeconds() / 2, p = I.getUTCFullYear() - 1980, p <<= 4, p |= I.getUTCMonth() + 1, p <<= 5, p |= I.getUTCDate(), v && (T = c(1, 1) + c(o(w), 4) + g, E += "up" + c(T.length, 2) + T), _ && (A = c(1, 1) + c(o(b), 4) + x, E += "uc" + c(A.length, 2) + A);
          var D = "";
          return D += "\n\0", D += c(N, 2), D += f.magic, D += c(h2, 2), D += c(p, 2), D += c(R.crc32, 4), D += c(R.compressedSize, 4), D += c(R.uncompressedSize, 4), D += c(w.length, 2), D += c(E.length, 2), { fileRecord: a.LOCAL_FILE_HEADER + D + w + E, dirRecord: a.CENTRAL_FILE_HEADER + c(L, 2) + D + c(b.length, 2) + "\0\0\0\0" + c(k, 4) + c(s2, 4) + w + E + b };
        }, l = function(e3) {
          return a.DATA_DESCRIPTOR + c(e3.crc32, 4) + c(e3.compressedSize, 4) + c(e3.uncompressedSize, 4);
        };
        function h(e3, t3, r3, n3) {
          s.call(this, "ZipFileWorker"), this.bytesWritten = 0, this.zipComment = t3, this.zipPlatform = r3, this.encodeFileName = n3, this.streamFiles = e3, this.accumulate = false, this.contentBuffer = [], this.dirRecords = [], this.currentSourceOffset = 0, this.entriesCount = 0, this.currentFile = null, this._sources = [];
        }
        n2.inherits(h, s), h.prototype.push = function(e3) {
          var t3 = e3.meta.percent || 0, r3 = this.entriesCount, n3 = this._sources.length;
          this.accumulate ? this.contentBuffer.push(e3) : (this.bytesWritten += e3.data.length, s.prototype.push.call(this, { data: e3.data, meta: { currentFile: this.currentFile, percent: r3 ? (t3 + 100 * (r3 - n3 - 1)) / r3 : 100 } }));
        }, h.prototype.openedSource = function(e3) {
          this.currentSourceOffset = this.bytesWritten, this.currentFile = e3.file.name;
          var t3 = this.streamFiles && !e3.file.dir;
          if (t3) {
            var r3 = u(e3, t3, false, this.currentSourceOffset, this.zipPlatform, this.encodeFileName);
            this.push({ data: r3.fileRecord, meta: { percent: 0 } });
          } else
            this.accumulate = true;
        }, h.prototype.closedSource = function(e3) {
          this.accumulate = false;
          var t3 = this.streamFiles && !e3.file.dir, r3 = u(e3, t3, true, this.currentSourceOffset, this.zipPlatform, this.encodeFileName);
          if (this.dirRecords.push(r3.dirRecord), t3)
            this.push({ data: l(e3), meta: { percent: 100 } });
          else
            for (this.push({ data: r3.fileRecord, meta: { percent: 0 } }); this.contentBuffer.length; )
              this.push(this.contentBuffer.shift());
          this.currentFile = null;
        }, h.prototype.flush = function() {
          for (var e3 = this.bytesWritten, t3 = 0; t3 < this.dirRecords.length; t3++)
            this.push({ data: this.dirRecords[t3], meta: { percent: 100 } });
          var r3 = this.bytesWritten - e3, s2 = function(e4, t4, r4, s3, i2) {
            var o2 = n2.transformTo("string", i2(s3));
            return a.CENTRAL_DIRECTORY_END + "\0\0\0\0" + c(e4, 2) + c(e4, 2) + c(t4, 4) + c(r4, 4) + c(o2.length, 2) + o2;
          }(this.dirRecords.length, r3, e3, this.zipComment, this.encodeFileName);
          this.push({ data: s2, meta: { percent: 100 } });
        }, h.prototype.prepareNextSource = function() {
          this.previous = this._sources.shift(), this.openedSource(this.previous.streamInfo), this.isPaused ? this.previous.pause() : this.previous.resume();
        }, h.prototype.registerPrevious = function(e3) {
          this._sources.push(e3);
          var t3 = this;
          return e3.on("data", function(e4) {
            t3.processChunk(e4);
          }), e3.on("end", function() {
            t3.closedSource(t3.previous.streamInfo), t3._sources.length ? t3.prepareNextSource() : t3.end();
          }), e3.on("error", function(e4) {
            t3.error(e4);
          }), this;
        }, h.prototype.resume = function() {
          return !!s.prototype.resume.call(this) && (!this.previous && this._sources.length ? (this.prepareNextSource(), true) : this.previous || this._sources.length || this.generatedError ? void 0 : (this.end(), true));
        }, h.prototype.error = function(e3) {
          var t3 = this._sources;
          if (!s.prototype.error.call(this, e3))
            return false;
          for (var r3 = 0; r3 < t3.length; r3++)
            try {
              t3[r3].error(e3);
            } catch (e4) {
            }
          return true;
        }, h.prototype.lock = function() {
          s.prototype.lock.call(this);
          for (var e3 = this._sources, t3 = 0; t3 < e3.length; t3++)
            e3[t3].lock();
        }, e2.exports = h;
      }, 7834: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(1678), s = r2(4979);
        t2.generateWorker = function(e3, t3, r3) {
          var i = new s(t3.streamFiles, r3, t3.platform, t3.encodeFileName), o = 0;
          try {
            e3.forEach(function(e4, r4) {
              o++;
              var s2 = function(e5, t4) {
                var r5 = e5 || t4, s3 = n2[r5];
                if (!s3)
                  throw new Error(r5 + " is not a valid compression method !");
                return s3;
              }(r4.options.compression, t3.compression), a = r4.options.compressionOptions || t3.compressionOptions || {}, c = r4.dir, u = r4.date;
              r4._compressWorker(s2, a).withStreamInfo("file", { name: e4, dir: c, date: u, comment: r4.comment || "", unixPermissions: r4.unixPermissions, dosPermissions: r4.dosPermissions }).pipe(i);
            }), i.entriesCount = o;
          } catch (e4) {
            i.error(e4);
          }
          return i;
        };
      }, 6085: (e2, t2, r2) => {
        "use strict";
        function n2() {
          if (!(this instanceof n2))
            return new n2();
          if (arguments.length)
            throw new Error("The constructor with parameters has been removed in JSZip 3.0, please check the upgrade guide.");
          this.files = /* @__PURE__ */ Object.create(null), this.comment = null, this.root = "", this.clone = function() {
            var e3 = new n2();
            for (var t3 in this)
              "function" != typeof this[t3] && (e3[t3] = this[t3]);
            return e3;
          };
        }
        n2.prototype = r2(7132), n2.prototype.loadAsync = r2(1062), n2.support = r2(3790), n2.defaults = r2(6032), n2.version = "3.10.1", n2.loadAsync = function(e3, t3) {
          return new n2().loadAsync(e3, t3);
        }, n2.external = r2(8565), e2.exports = n2;
      }, 1062: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(8910), s = r2(8565), i = r2(3600), o = r2(6624), a = r2(2541), c = r2(2182);
        function u(e3) {
          return new s.Promise(function(t3, r3) {
            var n3 = e3.decompressed.getContentWorker().pipe(new a());
            n3.on("error", function(e4) {
              r3(e4);
            }).on("end", function() {
              n3.streamInfo.crc32 !== e3.decompressed.crc32 ? r3(new Error("Corrupted zip : CRC32 mismatch")) : t3();
            }).resume();
          });
        }
        e2.exports = function(e3, t3) {
          var r3 = this;
          return t3 = n2.extend(t3 || {}, { base64: false, checkCRC32: false, optimizedBinaryString: false, createFolders: false, decodeFileName: i.utf8decode }), c.isNode && c.isStream(e3) ? s.Promise.reject(new Error("JSZip can't accept a stream when loading a zip file.")) : n2.prepareContent("the loaded zip file", e3, true, t3.optimizedBinaryString, t3.base64).then(function(e4) {
            var r4 = new o(t3);
            return r4.load(e4), r4;
          }).then(function(e4) {
            var r4 = [s.Promise.resolve(e4)], n3 = e4.files;
            if (t3.checkCRC32)
              for (var i2 = 0; i2 < n3.length; i2++)
                r4.push(u(n3[i2]));
            return s.Promise.all(r4);
          }).then(function(e4) {
            for (var s2 = e4.shift(), i2 = s2.files, o2 = 0; o2 < i2.length; o2++) {
              var a2 = i2[o2], c2 = a2.fileNameStr, u2 = n2.resolve(a2.fileNameStr);
              r3.file(u2, a2.decompressed, { binary: true, optimizedBinaryString: true, date: a2.date, dir: a2.dir, comment: a2.fileCommentStr.length ? a2.fileCommentStr : null, unixPermissions: a2.unixPermissions, dosPermissions: a2.dosPermissions, createFolders: t3.createFolders }), a2.dir || (r3.file(u2).unsafeOriginalName = c2);
            }
            return s2.zipComment.length && (r3.comment = s2.zipComment), r3;
          });
        };
      }, 2182: (e2) => {
        "use strict";
        e2.exports = { isNode: "undefined" != typeof Buffer, newBufferFrom: function(e3, t2) {
          if (Buffer.from && Buffer.from !== Uint8Array.from)
            return Buffer.from(e3, t2);
          if ("number" == typeof e3)
            throw new Error('The "data" argument must not be a number');
          return new Buffer(e3, t2);
        }, allocBuffer: function(e3) {
          if (Buffer.alloc)
            return Buffer.alloc(e3);
          var t2 = new Buffer(e3);
          return t2.fill(0), t2;
        }, isBuffer: function(e3) {
          return Buffer.isBuffer(e3);
        }, isStream: function(e3) {
          return e3 && "function" == typeof e3.on && "function" == typeof e3.pause && "function" == typeof e3.resume;
        } };
      }, 660: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(8910), s = r2(3718);
        function i(e3, t3) {
          s.call(this, "Nodejs stream input adapter for " + e3), this._upstreamEnded = false, this._bindStream(t3);
        }
        n2.inherits(i, s), i.prototype._bindStream = function(e3) {
          var t3 = this;
          this._stream = e3, e3.pause(), e3.on("data", function(e4) {
            t3.push({ data: e4, meta: { percent: 0 } });
          }).on("error", function(e4) {
            t3.isPaused ? this.generatedError = e4 : t3.error(e4);
          }).on("end", function() {
            t3.isPaused ? t3._upstreamEnded = true : t3.end();
          });
        }, i.prototype.pause = function() {
          return !!s.prototype.pause.call(this) && (this._stream.pause(), true);
        }, i.prototype.resume = function() {
          return !!s.prototype.resume.call(this) && (this._upstreamEnded ? this.end() : this._stream.resume(), true);
        }, e2.exports = i;
      }, 1220: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(749).Readable;
        function s(e3, t3, r3) {
          n2.call(this, t3), this._helper = e3;
          var s2 = this;
          e3.on("data", function(e4, t4) {
            s2.push(e4) || s2._helper.pause(), r3 && r3(t4);
          }).on("error", function(e4) {
            s2.emit("error", e4);
          }).on("end", function() {
            s2.push(null);
          });
        }
        r2(8910).inherits(s, n2), s.prototype._read = function() {
          this._helper.resume();
        }, e2.exports = s;
      }, 7132: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(3600), s = r2(8910), i = r2(3718), o = r2(1285), a = r2(6032), c = r2(7326), u = r2(6859), l = r2(7834), h = r2(2182), p = r2(660), d = function(e3, t3, r3) {
          var n3, o2 = s.getTypeOf(t3), l2 = s.extend(r3 || {}, a);
          l2.date = l2.date || /* @__PURE__ */ new Date(), null !== l2.compression && (l2.compression = l2.compression.toUpperCase()), "string" == typeof l2.unixPermissions && (l2.unixPermissions = parseInt(l2.unixPermissions, 8)), l2.unixPermissions && 16384 & l2.unixPermissions && (l2.dir = true), l2.dosPermissions && 16 & l2.dosPermissions && (l2.dir = true), l2.dir && (e3 = m(e3)), l2.createFolders && (n3 = f(e3)) && w.call(this, n3, true);
          var d2 = "string" === o2 && false === l2.binary && false === l2.base64;
          r3 && void 0 !== r3.binary || (l2.binary = !d2), (t3 instanceof c && 0 === t3.uncompressedSize || l2.dir || !t3 || 0 === t3.length) && (l2.base64 = false, l2.binary = true, t3 = "", l2.compression = "STORE", o2 = "string");
          var g2;
          g2 = t3 instanceof c || t3 instanceof i ? t3 : h.isNode && h.isStream(t3) ? new p(e3, t3) : s.prepareContent(e3, t3, l2.binary, l2.optimizedBinaryString, l2.base64);
          var y2 = new u(e3, g2, l2);
          this.files[e3] = y2;
        }, f = function(e3) {
          "/" === e3.slice(-1) && (e3 = e3.substring(0, e3.length - 1));
          var t3 = e3.lastIndexOf("/");
          return t3 > 0 ? e3.substring(0, t3) : "";
        }, m = function(e3) {
          return "/" !== e3.slice(-1) && (e3 += "/"), e3;
        }, w = function(e3, t3) {
          return t3 = void 0 !== t3 ? t3 : a.createFolders, e3 = m(e3), this.files[e3] || d.call(this, e3, null, { dir: true, createFolders: t3 }), this.files[e3];
        };
        function g(e3) {
          return "[object RegExp]" === Object.prototype.toString.call(e3);
        }
        var y = { load: function() {
          throw new Error("This method has been removed in JSZip 3.0, please check the upgrade guide.");
        }, forEach: function(e3) {
          var t3, r3, n3;
          for (t3 in this.files)
            n3 = this.files[t3], (r3 = t3.slice(this.root.length, t3.length)) && t3.slice(0, this.root.length) === this.root && e3(r3, n3);
        }, filter: function(e3) {
          var t3 = [];
          return this.forEach(function(r3, n3) {
            e3(r3, n3) && t3.push(n3);
          }), t3;
        }, file: function(e3, t3, r3) {
          if (1 === arguments.length) {
            if (g(e3)) {
              var n3 = e3;
              return this.filter(function(e4, t4) {
                return !t4.dir && n3.test(e4);
              });
            }
            var s2 = this.files[this.root + e3];
            return s2 && !s2.dir ? s2 : null;
          }
          return e3 = this.root + e3, d.call(this, e3, t3, r3), this;
        }, folder: function(e3) {
          if (!e3)
            return this;
          if (g(e3))
            return this.filter(function(t4, r4) {
              return r4.dir && e3.test(t4);
            });
          var t3 = this.root + e3, r3 = w.call(this, t3), n3 = this.clone();
          return n3.root = r3.name, n3;
        }, remove: function(e3) {
          e3 = this.root + e3;
          var t3 = this.files[e3];
          if (t3 || ("/" !== e3.slice(-1) && (e3 += "/"), t3 = this.files[e3]), t3 && !t3.dir)
            delete this.files[e3];
          else
            for (var r3 = this.filter(function(t4, r4) {
              return r4.name.slice(0, e3.length) === e3;
            }), n3 = 0; n3 < r3.length; n3++)
              delete this.files[r3[n3].name];
          return this;
        }, generate: function() {
          throw new Error("This method has been removed in JSZip 3.0, please check the upgrade guide.");
        }, generateInternalStream: function(e3) {
          var t3, r3 = {};
          try {
            if ((r3 = s.extend(e3 || {}, { streamFiles: false, compression: "STORE", compressionOptions: null, type: "", platform: "DOS", comment: null, mimeType: "application/zip", encodeFileName: n2.utf8encode })).type = r3.type.toLowerCase(), r3.compression = r3.compression.toUpperCase(), "binarystring" === r3.type && (r3.type = "string"), !r3.type)
              throw new Error("No output type specified.");
            s.checkSupport(r3.type), "darwin" !== r3.platform && "freebsd" !== r3.platform && "linux" !== r3.platform && "sunos" !== r3.platform || (r3.platform = "UNIX"), "win32" === r3.platform && (r3.platform = "DOS");
            var a2 = r3.comment || this.comment || "";
            t3 = l.generateWorker(this, r3, a2);
          } catch (e4) {
            (t3 = new i("error")).error(e4);
          }
          return new o(t3, r3.type || "string", r3.mimeType);
        }, generateAsync: function(e3, t3) {
          return this.generateInternalStream(e3).accumulate(t3);
        }, generateNodeStream: function(e3, t3) {
          return (e3 = e3 || {}).type || (e3.type = "nodebuffer"), this.generateInternalStream(e3).toNodejsStream(t3);
        } };
        e2.exports = y;
      }, 749: (e2, t2, r2) => {
        "use strict";
        e2.exports = r2(2830);
      }, 2370: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(8542);
        function s(e3) {
          n2.call(this, e3);
          for (var t3 = 0; t3 < this.data.length; t3++)
            e3[t3] = 255 & e3[t3];
        }
        r2(8910).inherits(s, n2), s.prototype.byteAt = function(e3) {
          return this.data[this.zero + e3];
        }, s.prototype.lastIndexOfSignature = function(e3) {
          for (var t3 = e3.charCodeAt(0), r3 = e3.charCodeAt(1), n3 = e3.charCodeAt(2), s2 = e3.charCodeAt(3), i = this.length - 4; i >= 0; --i)
            if (this.data[i] === t3 && this.data[i + 1] === r3 && this.data[i + 2] === n3 && this.data[i + 3] === s2)
              return i - this.zero;
          return -1;
        }, s.prototype.readAndCheckSignature = function(e3) {
          var t3 = e3.charCodeAt(0), r3 = e3.charCodeAt(1), n3 = e3.charCodeAt(2), s2 = e3.charCodeAt(3), i = this.readData(4);
          return t3 === i[0] && r3 === i[1] && n3 === i[2] && s2 === i[3];
        }, s.prototype.readData = function(e3) {
          if (this.checkOffset(e3), 0 === e3)
            return [];
          var t3 = this.data.slice(this.zero + this.index, this.zero + this.index + e3);
          return this.index += e3, t3;
        }, e2.exports = s;
      }, 8542: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(8910);
        function s(e3) {
          this.data = e3, this.length = e3.length, this.index = 0, this.zero = 0;
        }
        s.prototype = { checkOffset: function(e3) {
          this.checkIndex(this.index + e3);
        }, checkIndex: function(e3) {
          if (this.length < this.zero + e3 || e3 < 0)
            throw new Error("End of data reached (data length = " + this.length + ", asked index = " + e3 + "). Corrupted zip ?");
        }, setIndex: function(e3) {
          this.checkIndex(e3), this.index = e3;
        }, skip: function(e3) {
          this.setIndex(this.index + e3);
        }, byteAt: function() {
        }, readInt: function(e3) {
          var t3, r3 = 0;
          for (this.checkOffset(e3), t3 = this.index + e3 - 1; t3 >= this.index; t3--)
            r3 = (r3 << 8) + this.byteAt(t3);
          return this.index += e3, r3;
        }, readString: function(e3) {
          return n2.transformTo("string", this.readData(e3));
        }, readData: function() {
        }, lastIndexOfSignature: function() {
        }, readAndCheckSignature: function() {
        }, readDate: function() {
          var e3 = this.readInt(4);
          return new Date(Date.UTC(1980 + (e3 >> 25 & 127), (e3 >> 21 & 15) - 1, e3 >> 16 & 31, e3 >> 11 & 31, e3 >> 5 & 63, (31 & e3) << 1));
        } }, e2.exports = s;
      }, 9583: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(414);
        function s(e3) {
          n2.call(this, e3);
        }
        r2(8910).inherits(s, n2), s.prototype.readData = function(e3) {
          this.checkOffset(e3);
          var t3 = this.data.slice(this.zero + this.index, this.zero + this.index + e3);
          return this.index += e3, t3;
        }, e2.exports = s;
      }, 9226: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(8542);
        function s(e3) {
          n2.call(this, e3);
        }
        r2(8910).inherits(s, n2), s.prototype.byteAt = function(e3) {
          return this.data.charCodeAt(this.zero + e3);
        }, s.prototype.lastIndexOfSignature = function(e3) {
          return this.data.lastIndexOf(e3) - this.zero;
        }, s.prototype.readAndCheckSignature = function(e3) {
          return e3 === this.readData(4);
        }, s.prototype.readData = function(e3) {
          this.checkOffset(e3);
          var t3 = this.data.slice(this.zero + this.index, this.zero + this.index + e3);
          return this.index += e3, t3;
        }, e2.exports = s;
      }, 414: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(2370);
        function s(e3) {
          n2.call(this, e3);
        }
        r2(8910).inherits(s, n2), s.prototype.readData = function(e3) {
          if (this.checkOffset(e3), 0 === e3)
            return new Uint8Array(0);
          var t3 = this.data.subarray(this.zero + this.index, this.zero + this.index + e3);
          return this.index += e3, t3;
        }, e2.exports = s;
      }, 8435: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(8910), s = r2(3790), i = r2(2370), o = r2(9226), a = r2(9583), c = r2(414);
        e2.exports = function(e3) {
          var t3 = n2.getTypeOf(e3);
          return n2.checkSupport(t3), "string" !== t3 || s.uint8array ? "nodebuffer" === t3 ? new a(e3) : s.uint8array ? new c(n2.transformTo("uint8array", e3)) : new i(n2.transformTo("array", e3)) : new o(e3);
        };
      }, 1141: (e2, t2) => {
        "use strict";
        t2.LOCAL_FILE_HEADER = "PK", t2.CENTRAL_FILE_HEADER = "PK", t2.CENTRAL_DIRECTORY_END = "PK", t2.ZIP64_CENTRAL_DIRECTORY_LOCATOR = "PK\x07", t2.ZIP64_CENTRAL_DIRECTORY_END = "PK", t2.DATA_DESCRIPTOR = "PK\x07\b";
      }, 4293: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(3718), s = r2(8910);
        function i(e3) {
          n2.call(this, "ConvertWorker to " + e3), this.destType = e3;
        }
        s.inherits(i, n2), i.prototype.processChunk = function(e3) {
          this.push({ data: s.transformTo(this.destType, e3.data), meta: e3.meta });
        }, e2.exports = i;
      }, 2541: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(3718), s = r2(6988);
        function i() {
          n2.call(this, "Crc32Probe"), this.withStreamInfo("crc32", 0);
        }
        r2(8910).inherits(i, n2), i.prototype.processChunk = function(e3) {
          this.streamInfo.crc32 = s(e3.data, this.streamInfo.crc32 || 0), this.push(e3);
        }, e2.exports = i;
      }, 5977: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(8910), s = r2(3718);
        function i(e3) {
          s.call(this, "DataLengthProbe for " + e3), this.propName = e3, this.withStreamInfo(e3, 0);
        }
        n2.inherits(i, s), i.prototype.processChunk = function(e3) {
          if (e3) {
            var t3 = this.streamInfo[this.propName] || 0;
            this.streamInfo[this.propName] = t3 + e3.data.length;
          }
          s.prototype.processChunk.call(this, e3);
        }, e2.exports = i;
      }, 5301: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(8910), s = r2(3718);
        function i(e3) {
          s.call(this, "DataWorker");
          var t3 = this;
          this.dataIsReady = false, this.index = 0, this.max = 0, this.data = null, this.type = "", this._tickScheduled = false, e3.then(function(e4) {
            t3.dataIsReady = true, t3.data = e4, t3.max = e4 && e4.length || 0, t3.type = n2.getTypeOf(e4), t3.isPaused || t3._tickAndRepeat();
          }, function(e4) {
            t3.error(e4);
          });
        }
        n2.inherits(i, s), i.prototype.cleanUp = function() {
          s.prototype.cleanUp.call(this), this.data = null;
        }, i.prototype.resume = function() {
          return !!s.prototype.resume.call(this) && (!this._tickScheduled && this.dataIsReady && (this._tickScheduled = true, n2.delay(this._tickAndRepeat, [], this)), true);
        }, i.prototype._tickAndRepeat = function() {
          this._tickScheduled = false, this.isPaused || this.isFinished || (this._tick(), this.isFinished || (n2.delay(this._tickAndRepeat, [], this), this._tickScheduled = true));
        }, i.prototype._tick = function() {
          if (this.isPaused || this.isFinished)
            return false;
          var e3 = null, t3 = Math.min(this.max, this.index + 16384);
          if (this.index >= this.max)
            return this.end();
          switch (this.type) {
            case "string":
              e3 = this.data.substring(this.index, t3);
              break;
            case "uint8array":
              e3 = this.data.subarray(this.index, t3);
              break;
            case "array":
            case "nodebuffer":
              e3 = this.data.slice(this.index, t3);
          }
          return this.index = t3, this.push({ data: e3, meta: { percent: this.max ? this.index / this.max * 100 : 0 } });
        }, e2.exports = i;
      }, 3718: (e2) => {
        "use strict";
        function t2(e3) {
          this.name = e3 || "default", this.streamInfo = {}, this.generatedError = null, this.extraStreamInfo = {}, this.isPaused = true, this.isFinished = false, this.isLocked = false, this._listeners = { data: [], end: [], error: [] }, this.previous = null;
        }
        t2.prototype = { push: function(e3) {
          this.emit("data", e3);
        }, end: function() {
          if (this.isFinished)
            return false;
          this.flush();
          try {
            this.emit("end"), this.cleanUp(), this.isFinished = true;
          } catch (e3) {
            this.emit("error", e3);
          }
          return true;
        }, error: function(e3) {
          return !this.isFinished && (this.isPaused ? this.generatedError = e3 : (this.isFinished = true, this.emit("error", e3), this.previous && this.previous.error(e3), this.cleanUp()), true);
        }, on: function(e3, t3) {
          return this._listeners[e3].push(t3), this;
        }, cleanUp: function() {
          this.streamInfo = this.generatedError = this.extraStreamInfo = null, this._listeners = [];
        }, emit: function(e3, t3) {
          if (this._listeners[e3])
            for (var r2 = 0; r2 < this._listeners[e3].length; r2++)
              this._listeners[e3][r2].call(this, t3);
        }, pipe: function(e3) {
          return e3.registerPrevious(this);
        }, registerPrevious: function(e3) {
          if (this.isLocked)
            throw new Error("The stream '" + this + "' has already been used.");
          this.streamInfo = e3.streamInfo, this.mergeStreamInfo(), this.previous = e3;
          var t3 = this;
          return e3.on("data", function(e4) {
            t3.processChunk(e4);
          }), e3.on("end", function() {
            t3.end();
          }), e3.on("error", function(e4) {
            t3.error(e4);
          }), this;
        }, pause: function() {
          return !this.isPaused && !this.isFinished && (this.isPaused = true, this.previous && this.previous.pause(), true);
        }, resume: function() {
          if (!this.isPaused || this.isFinished)
            return false;
          this.isPaused = false;
          var e3 = false;
          return this.generatedError && (this.error(this.generatedError), e3 = true), this.previous && this.previous.resume(), !e3;
        }, flush: function() {
        }, processChunk: function(e3) {
          this.push(e3);
        }, withStreamInfo: function(e3, t3) {
          return this.extraStreamInfo[e3] = t3, this.mergeStreamInfo(), this;
        }, mergeStreamInfo: function() {
          for (var e3 in this.extraStreamInfo)
            Object.prototype.hasOwnProperty.call(this.extraStreamInfo, e3) && (this.streamInfo[e3] = this.extraStreamInfo[e3]);
        }, lock: function() {
          if (this.isLocked)
            throw new Error("The stream '" + this + "' has already been used.");
          this.isLocked = true, this.previous && this.previous.lock();
        }, toString: function() {
          var e3 = "Worker " + this.name;
          return this.previous ? this.previous + " -> " + e3 : e3;
        } }, e2.exports = t2;
      }, 1285: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(8910), s = r2(4293), i = r2(3718), o = r2(8458), a = r2(3790), c = r2(8565), u = null;
        if (a.nodestream)
          try {
            u = r2(1220);
          } catch (e3) {
          }
        function l(e3, t3, r3) {
          var o2 = t3;
          switch (t3) {
            case "blob":
            case "arraybuffer":
              o2 = "uint8array";
              break;
            case "base64":
              o2 = "string";
          }
          try {
            this._internalType = o2, this._outputType = t3, this._mimeType = r3, n2.checkSupport(o2), this._worker = e3.pipe(new s(o2)), e3.lock();
          } catch (e4) {
            this._worker = new i("error"), this._worker.error(e4);
          }
        }
        l.prototype = { accumulate: function(e3) {
          return t3 = this, r3 = e3, new c.Promise(function(e4, s2) {
            var i2 = [], a2 = t3._internalType, c2 = t3._outputType, u2 = t3._mimeType;
            t3.on("data", function(e5, t4) {
              i2.push(e5), r3 && r3(t4);
            }).on("error", function(e5) {
              i2 = [], s2(e5);
            }).on("end", function() {
              try {
                var t4 = function(e5, t5, r4) {
                  switch (e5) {
                    case "blob":
                      return n2.newBlob(n2.transformTo("arraybuffer", t5), r4);
                    case "base64":
                      return o.encode(t5);
                    default:
                      return n2.transformTo(e5, t5);
                  }
                }(c2, function(e5, t5) {
                  var r4, n3 = 0, s3 = null, i3 = 0;
                  for (r4 = 0; r4 < t5.length; r4++)
                    i3 += t5[r4].length;
                  switch (e5) {
                    case "string":
                      return t5.join("");
                    case "array":
                      return Array.prototype.concat.apply([], t5);
                    case "uint8array":
                      for (s3 = new Uint8Array(i3), r4 = 0; r4 < t5.length; r4++)
                        s3.set(t5[r4], n3), n3 += t5[r4].length;
                      return s3;
                    case "nodebuffer":
                      return Buffer.concat(t5);
                    default:
                      throw new Error("concat : unsupported type '" + e5 + "'");
                  }
                }(a2, i2), u2);
                e4(t4);
              } catch (e5) {
                s2(e5);
              }
              i2 = [];
            }).resume();
          });
          var t3, r3;
        }, on: function(e3, t3) {
          var r3 = this;
          return "data" === e3 ? this._worker.on(e3, function(e4) {
            t3.call(r3, e4.data, e4.meta);
          }) : this._worker.on(e3, function() {
            n2.delay(t3, arguments, r3);
          }), this;
        }, resume: function() {
          return n2.delay(this._worker.resume, [], this._worker), this;
        }, pause: function() {
          return this._worker.pause(), this;
        }, toNodejsStream: function(e3) {
          if (n2.checkSupport("nodestream"), "nodebuffer" !== this._outputType)
            throw new Error(this._outputType + " is not supported by this method");
          return new u(this, { objectMode: "nodebuffer" !== this._outputType }, e3);
        } }, e2.exports = l;
      }, 3790: (e2, t2, r2) => {
        "use strict";
        if (t2.base64 = true, t2.array = true, t2.string = true, t2.arraybuffer = "undefined" != typeof ArrayBuffer && "undefined" != typeof Uint8Array, t2.nodebuffer = "undefined" != typeof Buffer, t2.uint8array = "undefined" != typeof Uint8Array, "undefined" == typeof ArrayBuffer)
          t2.blob = false;
        else {
          var n2 = new ArrayBuffer(0);
          try {
            t2.blob = 0 === new Blob([n2], { type: "application/zip" }).size;
          } catch (e3) {
            try {
              var s = new (self.BlobBuilder || self.WebKitBlobBuilder || self.MozBlobBuilder || self.MSBlobBuilder)();
              s.append(n2), t2.blob = 0 === s.getBlob("application/zip").size;
            } catch (e4) {
              t2.blob = false;
            }
          }
        }
        try {
          t2.nodestream = !!r2(749).Readable;
        } catch (e3) {
          t2.nodestream = false;
        }
      }, 3600: (e2, t2, r2) => {
        "use strict";
        for (var n2 = r2(8910), s = r2(3790), i = r2(2182), o = r2(3718), a = new Array(256), c = 0; c < 256; c++)
          a[c] = c >= 252 ? 6 : c >= 248 ? 5 : c >= 240 ? 4 : c >= 224 ? 3 : c >= 192 ? 2 : 1;
        function u() {
          o.call(this, "utf-8 decode"), this.leftOver = null;
        }
        function l() {
          o.call(this, "utf-8 encode");
        }
        a[254] = a[254] = 1, t2.utf8encode = function(e3) {
          return s.nodebuffer ? i.newBufferFrom(e3, "utf-8") : function(e4) {
            var t3, r3, n3, i2, o2, a2 = e4.length, c2 = 0;
            for (i2 = 0; i2 < a2; i2++)
              55296 == (64512 & (r3 = e4.charCodeAt(i2))) && i2 + 1 < a2 && 56320 == (64512 & (n3 = e4.charCodeAt(i2 + 1))) && (r3 = 65536 + (r3 - 55296 << 10) + (n3 - 56320), i2++), c2 += r3 < 128 ? 1 : r3 < 2048 ? 2 : r3 < 65536 ? 3 : 4;
            for (t3 = s.uint8array ? new Uint8Array(c2) : new Array(c2), o2 = 0, i2 = 0; o2 < c2; i2++)
              55296 == (64512 & (r3 = e4.charCodeAt(i2))) && i2 + 1 < a2 && 56320 == (64512 & (n3 = e4.charCodeAt(i2 + 1))) && (r3 = 65536 + (r3 - 55296 << 10) + (n3 - 56320), i2++), r3 < 128 ? t3[o2++] = r3 : r3 < 2048 ? (t3[o2++] = 192 | r3 >>> 6, t3[o2++] = 128 | 63 & r3) : r3 < 65536 ? (t3[o2++] = 224 | r3 >>> 12, t3[o2++] = 128 | r3 >>> 6 & 63, t3[o2++] = 128 | 63 & r3) : (t3[o2++] = 240 | r3 >>> 18, t3[o2++] = 128 | r3 >>> 12 & 63, t3[o2++] = 128 | r3 >>> 6 & 63, t3[o2++] = 128 | 63 & r3);
            return t3;
          }(e3);
        }, t2.utf8decode = function(e3) {
          return s.nodebuffer ? n2.transformTo("nodebuffer", e3).toString("utf-8") : function(e4) {
            var t3, r3, s2, i2, o2 = e4.length, c2 = new Array(2 * o2);
            for (r3 = 0, t3 = 0; t3 < o2; )
              if ((s2 = e4[t3++]) < 128)
                c2[r3++] = s2;
              else if ((i2 = a[s2]) > 4)
                c2[r3++] = 65533, t3 += i2 - 1;
              else {
                for (s2 &= 2 === i2 ? 31 : 3 === i2 ? 15 : 7; i2 > 1 && t3 < o2; )
                  s2 = s2 << 6 | 63 & e4[t3++], i2--;
                i2 > 1 ? c2[r3++] = 65533 : s2 < 65536 ? c2[r3++] = s2 : (s2 -= 65536, c2[r3++] = 55296 | s2 >> 10 & 1023, c2[r3++] = 56320 | 1023 & s2);
              }
            return c2.length !== r3 && (c2.subarray ? c2 = c2.subarray(0, r3) : c2.length = r3), n2.applyFromCharCode(c2);
          }(e3 = n2.transformTo(s.uint8array ? "uint8array" : "array", e3));
        }, n2.inherits(u, o), u.prototype.processChunk = function(e3) {
          var r3 = n2.transformTo(s.uint8array ? "uint8array" : "array", e3.data);
          if (this.leftOver && this.leftOver.length) {
            if (s.uint8array) {
              var i2 = r3;
              (r3 = new Uint8Array(i2.length + this.leftOver.length)).set(this.leftOver, 0), r3.set(i2, this.leftOver.length);
            } else
              r3 = this.leftOver.concat(r3);
            this.leftOver = null;
          }
          var o2 = function(e4, t3) {
            var r4;
            for ((t3 = t3 || e4.length) > e4.length && (t3 = e4.length), r4 = t3 - 1; r4 >= 0 && 128 == (192 & e4[r4]); )
              r4--;
            return r4 < 0 || 0 === r4 ? t3 : r4 + a[e4[r4]] > t3 ? r4 : t3;
          }(r3), c2 = r3;
          o2 !== r3.length && (s.uint8array ? (c2 = r3.subarray(0, o2), this.leftOver = r3.subarray(o2, r3.length)) : (c2 = r3.slice(0, o2), this.leftOver = r3.slice(o2, r3.length))), this.push({ data: t2.utf8decode(c2), meta: e3.meta });
        }, u.prototype.flush = function() {
          this.leftOver && this.leftOver.length && (this.push({ data: t2.utf8decode(this.leftOver), meta: {} }), this.leftOver = null);
        }, t2.Utf8DecodeWorker = u, n2.inherits(l, o), l.prototype.processChunk = function(e3) {
          this.push({ data: t2.utf8encode(e3.data), meta: e3.meta });
        }, t2.Utf8EncodeWorker = l;
      }, 8910: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(3790), s = r2(8458), i = r2(2182), o = r2(8565);
        function a(e3) {
          return e3;
        }
        function c(e3, t3) {
          for (var r3 = 0; r3 < e3.length; ++r3)
            t3[r3] = 255 & e3.charCodeAt(r3);
          return t3;
        }
        r2(4889), t2.newBlob = function(e3, r3) {
          t2.checkSupport("blob");
          try {
            return new Blob([e3], { type: r3 });
          } catch (t3) {
            try {
              var n3 = new (self.BlobBuilder || self.WebKitBlobBuilder || self.MozBlobBuilder || self.MSBlobBuilder)();
              return n3.append(e3), n3.getBlob(r3);
            } catch (e4) {
              throw new Error("Bug : can't construct the Blob.");
            }
          }
        };
        var u = { stringifyByChunk: function(e3, t3, r3) {
          var n3 = [], s2 = 0, i2 = e3.length;
          if (i2 <= r3)
            return String.fromCharCode.apply(null, e3);
          for (; s2 < i2; )
            "array" === t3 || "nodebuffer" === t3 ? n3.push(String.fromCharCode.apply(null, e3.slice(s2, Math.min(s2 + r3, i2)))) : n3.push(String.fromCharCode.apply(null, e3.subarray(s2, Math.min(s2 + r3, i2)))), s2 += r3;
          return n3.join("");
        }, stringifyByChar: function(e3) {
          for (var t3 = "", r3 = 0; r3 < e3.length; r3++)
            t3 += String.fromCharCode(e3[r3]);
          return t3;
        }, applyCanBeUsed: { uint8array: function() {
          try {
            return n2.uint8array && 1 === String.fromCharCode.apply(null, new Uint8Array(1)).length;
          } catch (e3) {
            return false;
          }
        }(), nodebuffer: function() {
          try {
            return n2.nodebuffer && 1 === String.fromCharCode.apply(null, i.allocBuffer(1)).length;
          } catch (e3) {
            return false;
          }
        }() } };
        function l(e3) {
          var r3 = 65536, n3 = t2.getTypeOf(e3), s2 = true;
          if ("uint8array" === n3 ? s2 = u.applyCanBeUsed.uint8array : "nodebuffer" === n3 && (s2 = u.applyCanBeUsed.nodebuffer), s2)
            for (; r3 > 1; )
              try {
                return u.stringifyByChunk(e3, n3, r3);
              } catch (e4) {
                r3 = Math.floor(r3 / 2);
              }
          return u.stringifyByChar(e3);
        }
        function h(e3, t3) {
          for (var r3 = 0; r3 < e3.length; r3++)
            t3[r3] = e3[r3];
          return t3;
        }
        t2.applyFromCharCode = l;
        var p = {};
        p.string = { string: a, array: function(e3) {
          return c(e3, new Array(e3.length));
        }, arraybuffer: function(e3) {
          return p.string.uint8array(e3).buffer;
        }, uint8array: function(e3) {
          return c(e3, new Uint8Array(e3.length));
        }, nodebuffer: function(e3) {
          return c(e3, i.allocBuffer(e3.length));
        } }, p.array = { string: l, array: a, arraybuffer: function(e3) {
          return new Uint8Array(e3).buffer;
        }, uint8array: function(e3) {
          return new Uint8Array(e3);
        }, nodebuffer: function(e3) {
          return i.newBufferFrom(e3);
        } }, p.arraybuffer = { string: function(e3) {
          return l(new Uint8Array(e3));
        }, array: function(e3) {
          return h(new Uint8Array(e3), new Array(e3.byteLength));
        }, arraybuffer: a, uint8array: function(e3) {
          return new Uint8Array(e3);
        }, nodebuffer: function(e3) {
          return i.newBufferFrom(new Uint8Array(e3));
        } }, p.uint8array = { string: l, array: function(e3) {
          return h(e3, new Array(e3.length));
        }, arraybuffer: function(e3) {
          return e3.buffer;
        }, uint8array: a, nodebuffer: function(e3) {
          return i.newBufferFrom(e3);
        } }, p.nodebuffer = { string: l, array: function(e3) {
          return h(e3, new Array(e3.length));
        }, arraybuffer: function(e3) {
          return p.nodebuffer.uint8array(e3).buffer;
        }, uint8array: function(e3) {
          return h(e3, new Uint8Array(e3.length));
        }, nodebuffer: a }, t2.transformTo = function(e3, r3) {
          if (r3 || (r3 = ""), !e3)
            return r3;
          t2.checkSupport(e3);
          var n3 = t2.getTypeOf(r3);
          return p[n3][e3](r3);
        }, t2.resolve = function(e3) {
          for (var t3 = e3.split("/"), r3 = [], n3 = 0; n3 < t3.length; n3++) {
            var s2 = t3[n3];
            "." === s2 || "" === s2 && 0 !== n3 && n3 !== t3.length - 1 || (".." === s2 ? r3.pop() : r3.push(s2));
          }
          return r3.join("/");
        }, t2.getTypeOf = function(e3) {
          return "string" == typeof e3 ? "string" : "[object Array]" === Object.prototype.toString.call(e3) ? "array" : n2.nodebuffer && i.isBuffer(e3) ? "nodebuffer" : n2.uint8array && e3 instanceof Uint8Array ? "uint8array" : n2.arraybuffer && e3 instanceof ArrayBuffer ? "arraybuffer" : void 0;
        }, t2.checkSupport = function(e3) {
          if (!n2[e3.toLowerCase()])
            throw new Error(e3 + " is not supported by this platform");
        }, t2.MAX_VALUE_16BITS = 65535, t2.MAX_VALUE_32BITS = -1, t2.pretty = function(e3) {
          var t3, r3, n3 = "";
          for (r3 = 0; r3 < (e3 || "").length; r3++)
            n3 += "\\x" + ((t3 = e3.charCodeAt(r3)) < 16 ? "0" : "") + t3.toString(16).toUpperCase();
          return n3;
        }, t2.delay = function(e3, t3, r3) {
          setImmediate(function() {
            e3.apply(r3 || null, t3 || []);
          });
        }, t2.inherits = function(e3, t3) {
          var r3 = function() {
          };
          r3.prototype = t3.prototype, e3.prototype = new r3();
        }, t2.extend = function() {
          var e3, t3, r3 = {};
          for (e3 = 0; e3 < arguments.length; e3++)
            for (t3 in arguments[e3])
              Object.prototype.hasOwnProperty.call(arguments[e3], t3) && void 0 === r3[t3] && (r3[t3] = arguments[e3][t3]);
          return r3;
        }, t2.prepareContent = function(e3, r3, i2, a2, u2) {
          return o.Promise.resolve(r3).then(function(e4) {
            return n2.blob && (e4 instanceof Blob || -1 !== ["[object File]", "[object Blob]"].indexOf(Object.prototype.toString.call(e4))) && "undefined" != typeof FileReader ? new o.Promise(function(t3, r4) {
              var n3 = new FileReader();
              n3.onload = function(e5) {
                t3(e5.target.result);
              }, n3.onerror = function(e5) {
                r4(e5.target.error);
              }, n3.readAsArrayBuffer(e4);
            }) : e4;
          }).then(function(r4) {
            var l2, h2 = t2.getTypeOf(r4);
            return h2 ? ("arraybuffer" === h2 ? r4 = t2.transformTo("uint8array", r4) : "string" === h2 && (u2 ? r4 = s.decode(r4) : i2 && true !== a2 && (r4 = c(l2 = r4, n2.uint8array ? new Uint8Array(l2.length) : new Array(l2.length)))), r4) : o.Promise.reject(new Error("Can't read the data of '" + e3 + "'. Is it in a supported JavaScript type (String, Blob, ArrayBuffer, etc) ?"));
          });
        };
      }, 6624: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(8435), s = r2(8910), i = r2(1141), o = r2(9392), a = r2(3790);
        function c(e3) {
          this.files = [], this.loadOptions = e3;
        }
        c.prototype = { checkSignature: function(e3) {
          if (!this.reader.readAndCheckSignature(e3)) {
            this.reader.index -= 4;
            var t3 = this.reader.readString(4);
            throw new Error("Corrupted zip or bug: unexpected signature (" + s.pretty(t3) + ", expected " + s.pretty(e3) + ")");
          }
        }, isSignature: function(e3, t3) {
          var r3 = this.reader.index;
          this.reader.setIndex(e3);
          var n3 = this.reader.readString(4) === t3;
          return this.reader.setIndex(r3), n3;
        }, readBlockEndOfCentral: function() {
          this.diskNumber = this.reader.readInt(2), this.diskWithCentralDirStart = this.reader.readInt(2), this.centralDirRecordsOnThisDisk = this.reader.readInt(2), this.centralDirRecords = this.reader.readInt(2), this.centralDirSize = this.reader.readInt(4), this.centralDirOffset = this.reader.readInt(4), this.zipCommentLength = this.reader.readInt(2);
          var e3 = this.reader.readData(this.zipCommentLength), t3 = a.uint8array ? "uint8array" : "array", r3 = s.transformTo(t3, e3);
          this.zipComment = this.loadOptions.decodeFileName(r3);
        }, readBlockZip64EndOfCentral: function() {
          this.zip64EndOfCentralSize = this.reader.readInt(8), this.reader.skip(4), this.diskNumber = this.reader.readInt(4), this.diskWithCentralDirStart = this.reader.readInt(4), this.centralDirRecordsOnThisDisk = this.reader.readInt(8), this.centralDirRecords = this.reader.readInt(8), this.centralDirSize = this.reader.readInt(8), this.centralDirOffset = this.reader.readInt(8), this.zip64ExtensibleData = {};
          for (var e3, t3, r3, n3 = this.zip64EndOfCentralSize - 44; 0 < n3; )
            e3 = this.reader.readInt(2), t3 = this.reader.readInt(4), r3 = this.reader.readData(t3), this.zip64ExtensibleData[e3] = { id: e3, length: t3, value: r3 };
        }, readBlockZip64EndOfCentralLocator: function() {
          if (this.diskWithZip64CentralDirStart = this.reader.readInt(4), this.relativeOffsetEndOfZip64CentralDir = this.reader.readInt(8), this.disksCount = this.reader.readInt(4), this.disksCount > 1)
            throw new Error("Multi-volumes zip are not supported");
        }, readLocalFiles: function() {
          var e3, t3;
          for (e3 = 0; e3 < this.files.length; e3++)
            t3 = this.files[e3], this.reader.setIndex(t3.localHeaderOffset), this.checkSignature(i.LOCAL_FILE_HEADER), t3.readLocalPart(this.reader), t3.handleUTF8(), t3.processAttributes();
        }, readCentralDir: function() {
          var e3;
          for (this.reader.setIndex(this.centralDirOffset); this.reader.readAndCheckSignature(i.CENTRAL_FILE_HEADER); )
            (e3 = new o({ zip64: this.zip64 }, this.loadOptions)).readCentralPart(this.reader), this.files.push(e3);
          if (this.centralDirRecords !== this.files.length && 0 !== this.centralDirRecords && 0 === this.files.length)
            throw new Error("Corrupted zip or bug: expected " + this.centralDirRecords + " records in central dir, got " + this.files.length);
        }, readEndOfCentral: function() {
          var e3 = this.reader.lastIndexOfSignature(i.CENTRAL_DIRECTORY_END);
          if (e3 < 0)
            throw this.isSignature(0, i.LOCAL_FILE_HEADER) ? new Error("Corrupted zip: can't find end of central directory") : new Error("Can't find end of central directory : is this a zip file ? If it is, see https://stuk.github.io/jszip/documentation/howto/read_zip.html");
          this.reader.setIndex(e3);
          var t3 = e3;
          if (this.checkSignature(i.CENTRAL_DIRECTORY_END), this.readBlockEndOfCentral(), this.diskNumber === s.MAX_VALUE_16BITS || this.diskWithCentralDirStart === s.MAX_VALUE_16BITS || this.centralDirRecordsOnThisDisk === s.MAX_VALUE_16BITS || this.centralDirRecords === s.MAX_VALUE_16BITS || this.centralDirSize === s.MAX_VALUE_32BITS || this.centralDirOffset === s.MAX_VALUE_32BITS) {
            if (this.zip64 = true, (e3 = this.reader.lastIndexOfSignature(i.ZIP64_CENTRAL_DIRECTORY_LOCATOR)) < 0)
              throw new Error("Corrupted zip: can't find the ZIP64 end of central directory locator");
            if (this.reader.setIndex(e3), this.checkSignature(i.ZIP64_CENTRAL_DIRECTORY_LOCATOR), this.readBlockZip64EndOfCentralLocator(), !this.isSignature(this.relativeOffsetEndOfZip64CentralDir, i.ZIP64_CENTRAL_DIRECTORY_END) && (this.relativeOffsetEndOfZip64CentralDir = this.reader.lastIndexOfSignature(i.ZIP64_CENTRAL_DIRECTORY_END), this.relativeOffsetEndOfZip64CentralDir < 0))
              throw new Error("Corrupted zip: can't find the ZIP64 end of central directory");
            this.reader.setIndex(this.relativeOffsetEndOfZip64CentralDir), this.checkSignature(i.ZIP64_CENTRAL_DIRECTORY_END), this.readBlockZip64EndOfCentral();
          }
          var r3 = this.centralDirOffset + this.centralDirSize;
          this.zip64 && (r3 += 20, r3 += 12 + this.zip64EndOfCentralSize);
          var n3 = t3 - r3;
          if (n3 > 0)
            this.isSignature(t3, i.CENTRAL_FILE_HEADER) || (this.reader.zero = n3);
          else if (n3 < 0)
            throw new Error("Corrupted zip: missing " + Math.abs(n3) + " bytes.");
        }, prepareReader: function(e3) {
          this.reader = n2(e3);
        }, load: function(e3) {
          this.prepareReader(e3), this.readEndOfCentral(), this.readCentralDir(), this.readLocalFiles();
        } }, e2.exports = c;
      }, 9392: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(8435), s = r2(8910), i = r2(7326), o = r2(6988), a = r2(3600), c = r2(1678), u = r2(3790);
        function l(e3, t3) {
          this.options = e3, this.loadOptions = t3;
        }
        l.prototype = { isEncrypted: function() {
          return 1 == (1 & this.bitFlag);
        }, useUTF8: function() {
          return 2048 == (2048 & this.bitFlag);
        }, readLocalPart: function(e3) {
          var t3, r3;
          if (e3.skip(22), this.fileNameLength = e3.readInt(2), r3 = e3.readInt(2), this.fileName = e3.readData(this.fileNameLength), e3.skip(r3), -1 === this.compressedSize || -1 === this.uncompressedSize)
            throw new Error("Bug or corrupted zip : didn't get enough information from the central directory (compressedSize === -1 || uncompressedSize === -1)");
          if (null === (t3 = function(e4) {
            for (var t4 in c)
              if (Object.prototype.hasOwnProperty.call(c, t4) && c[t4].magic === e4)
                return c[t4];
            return null;
          }(this.compressionMethod)))
            throw new Error("Corrupted zip : compression " + s.pretty(this.compressionMethod) + " unknown (inner file : " + s.transformTo("string", this.fileName) + ")");
          this.decompressed = new i(this.compressedSize, this.uncompressedSize, this.crc32, t3, e3.readData(this.compressedSize));
        }, readCentralPart: function(e3) {
          this.versionMadeBy = e3.readInt(2), e3.skip(2), this.bitFlag = e3.readInt(2), this.compressionMethod = e3.readString(2), this.date = e3.readDate(), this.crc32 = e3.readInt(4), this.compressedSize = e3.readInt(4), this.uncompressedSize = e3.readInt(4);
          var t3 = e3.readInt(2);
          if (this.extraFieldsLength = e3.readInt(2), this.fileCommentLength = e3.readInt(2), this.diskNumberStart = e3.readInt(2), this.internalFileAttributes = e3.readInt(2), this.externalFileAttributes = e3.readInt(4), this.localHeaderOffset = e3.readInt(4), this.isEncrypted())
            throw new Error("Encrypted zip are not supported");
          e3.skip(t3), this.readExtraFields(e3), this.parseZIP64ExtraField(e3), this.fileComment = e3.readData(this.fileCommentLength);
        }, processAttributes: function() {
          this.unixPermissions = null, this.dosPermissions = null;
          var e3 = this.versionMadeBy >> 8;
          this.dir = !!(16 & this.externalFileAttributes), 0 === e3 && (this.dosPermissions = 63 & this.externalFileAttributes), 3 === e3 && (this.unixPermissions = this.externalFileAttributes >> 16 & 65535), this.dir || "/" !== this.fileNameStr.slice(-1) || (this.dir = true);
        }, parseZIP64ExtraField: function() {
          if (this.extraFields[1]) {
            var e3 = n2(this.extraFields[1].value);
            this.uncompressedSize === s.MAX_VALUE_32BITS && (this.uncompressedSize = e3.readInt(8)), this.compressedSize === s.MAX_VALUE_32BITS && (this.compressedSize = e3.readInt(8)), this.localHeaderOffset === s.MAX_VALUE_32BITS && (this.localHeaderOffset = e3.readInt(8)), this.diskNumberStart === s.MAX_VALUE_32BITS && (this.diskNumberStart = e3.readInt(4));
          }
        }, readExtraFields: function(e3) {
          var t3, r3, n3, s2 = e3.index + this.extraFieldsLength;
          for (this.extraFields || (this.extraFields = {}); e3.index + 4 < s2; )
            t3 = e3.readInt(2), r3 = e3.readInt(2), n3 = e3.readData(r3), this.extraFields[t3] = { id: t3, length: r3, value: n3 };
          e3.setIndex(s2);
        }, handleUTF8: function() {
          var e3 = u.uint8array ? "uint8array" : "array";
          if (this.useUTF8())
            this.fileNameStr = a.utf8decode(this.fileName), this.fileCommentStr = a.utf8decode(this.fileComment);
          else {
            var t3 = this.findExtraFieldUnicodePath();
            if (null !== t3)
              this.fileNameStr = t3;
            else {
              var r3 = s.transformTo(e3, this.fileName);
              this.fileNameStr = this.loadOptions.decodeFileName(r3);
            }
            var n3 = this.findExtraFieldUnicodeComment();
            if (null !== n3)
              this.fileCommentStr = n3;
            else {
              var i2 = s.transformTo(e3, this.fileComment);
              this.fileCommentStr = this.loadOptions.decodeFileName(i2);
            }
          }
        }, findExtraFieldUnicodePath: function() {
          var e3 = this.extraFields[28789];
          if (e3) {
            var t3 = n2(e3.value);
            return 1 !== t3.readInt(1) || o(this.fileName) !== t3.readInt(4) ? null : a.utf8decode(t3.readData(e3.length - 5));
          }
          return null;
        }, findExtraFieldUnicodeComment: function() {
          var e3 = this.extraFields[25461];
          if (e3) {
            var t3 = n2(e3.value);
            return 1 !== t3.readInt(1) || o(this.fileComment) !== t3.readInt(4) ? null : a.utf8decode(t3.readData(e3.length - 5));
          }
          return null;
        } }, e2.exports = l;
      }, 6859: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(1285), s = r2(5301), i = r2(3600), o = r2(7326), a = r2(3718), c = function(e3, t3, r3) {
          this.name = e3, this.dir = r3.dir, this.date = r3.date, this.comment = r3.comment, this.unixPermissions = r3.unixPermissions, this.dosPermissions = r3.dosPermissions, this._data = t3, this._dataBinary = r3.binary, this.options = { compression: r3.compression, compressionOptions: r3.compressionOptions };
        };
        c.prototype = { internalStream: function(e3) {
          var t3 = null, r3 = "string";
          try {
            if (!e3)
              throw new Error("No output type specified.");
            var s2 = "string" === (r3 = e3.toLowerCase()) || "text" === r3;
            "binarystring" !== r3 && "text" !== r3 || (r3 = "string"), t3 = this._decompressWorker();
            var o2 = !this._dataBinary;
            o2 && !s2 && (t3 = t3.pipe(new i.Utf8EncodeWorker())), !o2 && s2 && (t3 = t3.pipe(new i.Utf8DecodeWorker()));
          } catch (e4) {
            (t3 = new a("error")).error(e4);
          }
          return new n2(t3, r3, "");
        }, async: function(e3, t3) {
          return this.internalStream(e3).accumulate(t3);
        }, nodeStream: function(e3, t3) {
          return this.internalStream(e3 || "nodebuffer").toNodejsStream(t3);
        }, _compressWorker: function(e3, t3) {
          if (this._data instanceof o && this._data.compression.magic === e3.magic)
            return this._data.getCompressedWorker();
          var r3 = this._decompressWorker();
          return this._dataBinary || (r3 = r3.pipe(new i.Utf8EncodeWorker())), o.createWorkerFrom(r3, e3, t3);
        }, _decompressWorker: function() {
          return this._data instanceof o ? this._data.getContentWorker() : this._data instanceof a ? this._data : new s(this._data);
        } };
        for (var u = ["asText", "asBinary", "asNodeBuffer", "asUint8Array", "asArrayBuffer"], l = function() {
          throw new Error("This method has been removed in JSZip 3.0, please check the upgrade guide.");
        }, h = 0; h < u.length; h++)
          c.prototype[u[h]] = l;
        e2.exports = c;
      }, 3389: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(5705);
        function s() {
        }
        var i = {}, o = ["REJECTED"], a = ["FULFILLED"], c = ["PENDING"];
        function u(e3) {
          if ("function" != typeof e3)
            throw new TypeError("resolver must be a function");
          this.state = c, this.queue = [], this.outcome = void 0, e3 !== s && d(this, e3);
        }
        function l(e3, t3, r3) {
          this.promise = e3, "function" == typeof t3 && (this.onFulfilled = t3, this.callFulfilled = this.otherCallFulfilled), "function" == typeof r3 && (this.onRejected = r3, this.callRejected = this.otherCallRejected);
        }
        function h(e3, t3, r3) {
          n2(function() {
            var n3;
            try {
              n3 = t3(r3);
            } catch (t4) {
              return i.reject(e3, t4);
            }
            n3 === e3 ? i.reject(e3, new TypeError("Cannot resolve promise with itself")) : i.resolve(e3, n3);
          });
        }
        function p(e3) {
          var t3 = e3 && e3.then;
          if (e3 && ("object" == typeof e3 || "function" == typeof e3) && "function" == typeof t3)
            return function() {
              t3.apply(e3, arguments);
            };
        }
        function d(e3, t3) {
          var r3 = false;
          function n3(t4) {
            r3 || (r3 = true, i.reject(e3, t4));
          }
          function s2(t4) {
            r3 || (r3 = true, i.resolve(e3, t4));
          }
          var o2 = f(function() {
            t3(s2, n3);
          });
          "error" === o2.status && n3(o2.value);
        }
        function f(e3, t3) {
          var r3 = {};
          try {
            r3.value = e3(t3), r3.status = "success";
          } catch (e4) {
            r3.status = "error", r3.value = e4;
          }
          return r3;
        }
        e2.exports = u, u.prototype.finally = function(e3) {
          if ("function" != typeof e3)
            return this;
          var t3 = this.constructor;
          return this.then(function(r3) {
            return t3.resolve(e3()).then(function() {
              return r3;
            });
          }, function(r3) {
            return t3.resolve(e3()).then(function() {
              throw r3;
            });
          });
        }, u.prototype.catch = function(e3) {
          return this.then(null, e3);
        }, u.prototype.then = function(e3, t3) {
          if ("function" != typeof e3 && this.state === a || "function" != typeof t3 && this.state === o)
            return this;
          var r3 = new this.constructor(s);
          return this.state !== c ? h(r3, this.state === a ? e3 : t3, this.outcome) : this.queue.push(new l(r3, e3, t3)), r3;
        }, l.prototype.callFulfilled = function(e3) {
          i.resolve(this.promise, e3);
        }, l.prototype.otherCallFulfilled = function(e3) {
          h(this.promise, this.onFulfilled, e3);
        }, l.prototype.callRejected = function(e3) {
          i.reject(this.promise, e3);
        }, l.prototype.otherCallRejected = function(e3) {
          h(this.promise, this.onRejected, e3);
        }, i.resolve = function(e3, t3) {
          var r3 = f(p, t3);
          if ("error" === r3.status)
            return i.reject(e3, r3.value);
          var n3 = r3.value;
          if (n3)
            d(e3, n3);
          else {
            e3.state = a, e3.outcome = t3;
            for (var s2 = -1, o2 = e3.queue.length; ++s2 < o2; )
              e3.queue[s2].callFulfilled(t3);
          }
          return e3;
        }, i.reject = function(e3, t3) {
          e3.state = o, e3.outcome = t3;
          for (var r3 = -1, n3 = e3.queue.length; ++r3 < n3; )
            e3.queue[r3].callRejected(t3);
          return e3;
        }, u.resolve = function(e3) {
          return e3 instanceof this ? e3 : i.resolve(new this(s), e3);
        }, u.reject = function(e3) {
          var t3 = new this(s);
          return i.reject(t3, e3);
        }, u.all = function(e3) {
          var t3 = this;
          if ("[object Array]" !== Object.prototype.toString.call(e3))
            return this.reject(new TypeError("must be an array"));
          var r3 = e3.length, n3 = false;
          if (!r3)
            return this.resolve([]);
          for (var o2 = new Array(r3), a2 = 0, c2 = -1, u2 = new this(s); ++c2 < r3; )
            l2(e3[c2], c2);
          return u2;
          function l2(e4, s2) {
            t3.resolve(e4).then(function(e5) {
              o2[s2] = e5, ++a2 !== r3 || n3 || (n3 = true, i.resolve(u2, o2));
            }, function(e5) {
              n3 || (n3 = true, i.reject(u2, e5));
            });
          }
        }, u.race = function(e3) {
          if ("[object Array]" !== Object.prototype.toString.call(e3))
            return this.reject(new TypeError("must be an array"));
          var t3 = e3.length, r3 = false;
          if (!t3)
            return this.resolve([]);
          for (var n3, o2 = -1, a2 = new this(s); ++o2 < t3; )
            n3 = e3[o2], this.resolve(n3).then(function(e4) {
              r3 || (r3 = true, i.resolve(a2, e4));
            }, function(e4) {
              r3 || (r3 = true, i.reject(a2, e4));
            });
          return a2;
        };
      }, 9591: (e2, t2, r2) => {
        "use strict";
        var n2 = {};
        (0, r2(4236).assign)(n2, r2(4555), r2(8843), r2(1619)), e2.exports = n2;
      }, 4555: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(405), s = r2(4236), i = r2(9373), o = r2(8898), a = r2(2292), c = Object.prototype.toString;
        function u(e3) {
          if (!(this instanceof u))
            return new u(e3);
          this.options = s.assign({ level: -1, method: 8, chunkSize: 16384, windowBits: 15, memLevel: 8, strategy: 0, to: "" }, e3 || {});
          var t3 = this.options;
          t3.raw && t3.windowBits > 0 ? t3.windowBits = -t3.windowBits : t3.gzip && t3.windowBits > 0 && t3.windowBits < 16 && (t3.windowBits += 16), this.err = 0, this.msg = "", this.ended = false, this.chunks = [], this.strm = new a(), this.strm.avail_out = 0;
          var r3 = n2.deflateInit2(this.strm, t3.level, t3.method, t3.windowBits, t3.memLevel, t3.strategy);
          if (0 !== r3)
            throw new Error(o[r3]);
          if (t3.header && n2.deflateSetHeader(this.strm, t3.header), t3.dictionary) {
            var l2;
            if (l2 = "string" == typeof t3.dictionary ? i.string2buf(t3.dictionary) : "[object ArrayBuffer]" === c.call(t3.dictionary) ? new Uint8Array(t3.dictionary) : t3.dictionary, 0 !== (r3 = n2.deflateSetDictionary(this.strm, l2)))
              throw new Error(o[r3]);
            this._dict_set = true;
          }
        }
        function l(e3, t3) {
          var r3 = new u(t3);
          if (r3.push(e3, true), r3.err)
            throw r3.msg || o[r3.err];
          return r3.result;
        }
        u.prototype.push = function(e3, t3) {
          var r3, o2, a2 = this.strm, u2 = this.options.chunkSize;
          if (this.ended)
            return false;
          o2 = t3 === ~~t3 ? t3 : true === t3 ? 4 : 0, "string" == typeof e3 ? a2.input = i.string2buf(e3) : "[object ArrayBuffer]" === c.call(e3) ? a2.input = new Uint8Array(e3) : a2.input = e3, a2.next_in = 0, a2.avail_in = a2.input.length;
          do {
            if (0 === a2.avail_out && (a2.output = new s.Buf8(u2), a2.next_out = 0, a2.avail_out = u2), 1 !== (r3 = n2.deflate(a2, o2)) && 0 !== r3)
              return this.onEnd(r3), this.ended = true, false;
            0 !== a2.avail_out && (0 !== a2.avail_in || 4 !== o2 && 2 !== o2) || ("string" === this.options.to ? this.onData(i.buf2binstring(s.shrinkBuf(a2.output, a2.next_out))) : this.onData(s.shrinkBuf(a2.output, a2.next_out)));
          } while ((a2.avail_in > 0 || 0 === a2.avail_out) && 1 !== r3);
          return 4 === o2 ? (r3 = n2.deflateEnd(this.strm), this.onEnd(r3), this.ended = true, 0 === r3) : 2 !== o2 || (this.onEnd(0), a2.avail_out = 0, true);
        }, u.prototype.onData = function(e3) {
          this.chunks.push(e3);
        }, u.prototype.onEnd = function(e3) {
          0 === e3 && ("string" === this.options.to ? this.result = this.chunks.join("") : this.result = s.flattenChunks(this.chunks)), this.chunks = [], this.err = e3, this.msg = this.strm.msg;
        }, t2.Deflate = u, t2.deflate = l, t2.deflateRaw = function(e3, t3) {
          return (t3 = t3 || {}).raw = true, l(e3, t3);
        }, t2.gzip = function(e3, t3) {
          return (t3 = t3 || {}).gzip = true, l(e3, t3);
        };
      }, 8843: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(7948), s = r2(4236), i = r2(9373), o = r2(1619), a = r2(8898), c = r2(2292), u = r2(2401), l = Object.prototype.toString;
        function h(e3) {
          if (!(this instanceof h))
            return new h(e3);
          this.options = s.assign({ chunkSize: 16384, windowBits: 0, to: "" }, e3 || {});
          var t3 = this.options;
          t3.raw && t3.windowBits >= 0 && t3.windowBits < 16 && (t3.windowBits = -t3.windowBits, 0 === t3.windowBits && (t3.windowBits = -15)), !(t3.windowBits >= 0 && t3.windowBits < 16) || e3 && e3.windowBits || (t3.windowBits += 32), t3.windowBits > 15 && t3.windowBits < 48 && 0 == (15 & t3.windowBits) && (t3.windowBits |= 15), this.err = 0, this.msg = "", this.ended = false, this.chunks = [], this.strm = new c(), this.strm.avail_out = 0;
          var r3 = n2.inflateInit2(this.strm, t3.windowBits);
          if (r3 !== o.Z_OK)
            throw new Error(a[r3]);
          if (this.header = new u(), n2.inflateGetHeader(this.strm, this.header), t3.dictionary && ("string" == typeof t3.dictionary ? t3.dictionary = i.string2buf(t3.dictionary) : "[object ArrayBuffer]" === l.call(t3.dictionary) && (t3.dictionary = new Uint8Array(t3.dictionary)), t3.raw && (r3 = n2.inflateSetDictionary(this.strm, t3.dictionary)) !== o.Z_OK))
            throw new Error(a[r3]);
        }
        function p(e3, t3) {
          var r3 = new h(t3);
          if (r3.push(e3, true), r3.err)
            throw r3.msg || a[r3.err];
          return r3.result;
        }
        h.prototype.push = function(e3, t3) {
          var r3, a2, c2, u2, h2, p2 = this.strm, d = this.options.chunkSize, f = this.options.dictionary, m = false;
          if (this.ended)
            return false;
          a2 = t3 === ~~t3 ? t3 : true === t3 ? o.Z_FINISH : o.Z_NO_FLUSH, "string" == typeof e3 ? p2.input = i.binstring2buf(e3) : "[object ArrayBuffer]" === l.call(e3) ? p2.input = new Uint8Array(e3) : p2.input = e3, p2.next_in = 0, p2.avail_in = p2.input.length;
          do {
            if (0 === p2.avail_out && (p2.output = new s.Buf8(d), p2.next_out = 0, p2.avail_out = d), (r3 = n2.inflate(p2, o.Z_NO_FLUSH)) === o.Z_NEED_DICT && f && (r3 = n2.inflateSetDictionary(this.strm, f)), r3 === o.Z_BUF_ERROR && true === m && (r3 = o.Z_OK, m = false), r3 !== o.Z_STREAM_END && r3 !== o.Z_OK)
              return this.onEnd(r3), this.ended = true, false;
            p2.next_out && (0 !== p2.avail_out && r3 !== o.Z_STREAM_END && (0 !== p2.avail_in || a2 !== o.Z_FINISH && a2 !== o.Z_SYNC_FLUSH) || ("string" === this.options.to ? (c2 = i.utf8border(p2.output, p2.next_out), u2 = p2.next_out - c2, h2 = i.buf2string(p2.output, c2), p2.next_out = u2, p2.avail_out = d - u2, u2 && s.arraySet(p2.output, p2.output, c2, u2, 0), this.onData(h2)) : this.onData(s.shrinkBuf(p2.output, p2.next_out)))), 0 === p2.avail_in && 0 === p2.avail_out && (m = true);
          } while ((p2.avail_in > 0 || 0 === p2.avail_out) && r3 !== o.Z_STREAM_END);
          return r3 === o.Z_STREAM_END && (a2 = o.Z_FINISH), a2 === o.Z_FINISH ? (r3 = n2.inflateEnd(this.strm), this.onEnd(r3), this.ended = true, r3 === o.Z_OK) : a2 !== o.Z_SYNC_FLUSH || (this.onEnd(o.Z_OK), p2.avail_out = 0, true);
        }, h.prototype.onData = function(e3) {
          this.chunks.push(e3);
        }, h.prototype.onEnd = function(e3) {
          e3 === o.Z_OK && ("string" === this.options.to ? this.result = this.chunks.join("") : this.result = s.flattenChunks(this.chunks)), this.chunks = [], this.err = e3, this.msg = this.strm.msg;
        }, t2.Inflate = h, t2.inflate = p, t2.inflateRaw = function(e3, t3) {
          return (t3 = t3 || {}).raw = true, p(e3, t3);
        }, t2.ungzip = p;
      }, 4236: (e2, t2) => {
        "use strict";
        var r2 = "undefined" != typeof Uint8Array && "undefined" != typeof Uint16Array && "undefined" != typeof Int32Array;
        function n2(e3, t3) {
          return Object.prototype.hasOwnProperty.call(e3, t3);
        }
        t2.assign = function(e3) {
          for (var t3 = Array.prototype.slice.call(arguments, 1); t3.length; ) {
            var r3 = t3.shift();
            if (r3) {
              if ("object" != typeof r3)
                throw new TypeError(r3 + "must be non-object");
              for (var s2 in r3)
                n2(r3, s2) && (e3[s2] = r3[s2]);
            }
          }
          return e3;
        }, t2.shrinkBuf = function(e3, t3) {
          return e3.length === t3 ? e3 : e3.subarray ? e3.subarray(0, t3) : (e3.length = t3, e3);
        };
        var s = { arraySet: function(e3, t3, r3, n3, s2) {
          if (t3.subarray && e3.subarray)
            e3.set(t3.subarray(r3, r3 + n3), s2);
          else
            for (var i2 = 0; i2 < n3; i2++)
              e3[s2 + i2] = t3[r3 + i2];
        }, flattenChunks: function(e3) {
          var t3, r3, n3, s2, i2, o;
          for (n3 = 0, t3 = 0, r3 = e3.length; t3 < r3; t3++)
            n3 += e3[t3].length;
          for (o = new Uint8Array(n3), s2 = 0, t3 = 0, r3 = e3.length; t3 < r3; t3++)
            i2 = e3[t3], o.set(i2, s2), s2 += i2.length;
          return o;
        } }, i = { arraySet: function(e3, t3, r3, n3, s2) {
          for (var i2 = 0; i2 < n3; i2++)
            e3[s2 + i2] = t3[r3 + i2];
        }, flattenChunks: function(e3) {
          return [].concat.apply([], e3);
        } };
        t2.setTyped = function(e3) {
          e3 ? (t2.Buf8 = Uint8Array, t2.Buf16 = Uint16Array, t2.Buf32 = Int32Array, t2.assign(t2, s)) : (t2.Buf8 = Array, t2.Buf16 = Array, t2.Buf32 = Array, t2.assign(t2, i));
        }, t2.setTyped(r2);
      }, 9373: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(4236), s = true, i = true;
        try {
          String.fromCharCode.apply(null, [0]);
        } catch (e3) {
          s = false;
        }
        try {
          String.fromCharCode.apply(null, new Uint8Array(1));
        } catch (e3) {
          i = false;
        }
        for (var o = new n2.Buf8(256), a = 0; a < 256; a++)
          o[a] = a >= 252 ? 6 : a >= 248 ? 5 : a >= 240 ? 4 : a >= 224 ? 3 : a >= 192 ? 2 : 1;
        function c(e3, t3) {
          if (t3 < 65534 && (e3.subarray && i || !e3.subarray && s))
            return String.fromCharCode.apply(null, n2.shrinkBuf(e3, t3));
          for (var r3 = "", o2 = 0; o2 < t3; o2++)
            r3 += String.fromCharCode(e3[o2]);
          return r3;
        }
        o[254] = o[254] = 1, t2.string2buf = function(e3) {
          var t3, r3, s2, i2, o2, a2 = e3.length, c2 = 0;
          for (i2 = 0; i2 < a2; i2++)
            55296 == (64512 & (r3 = e3.charCodeAt(i2))) && i2 + 1 < a2 && 56320 == (64512 & (s2 = e3.charCodeAt(i2 + 1))) && (r3 = 65536 + (r3 - 55296 << 10) + (s2 - 56320), i2++), c2 += r3 < 128 ? 1 : r3 < 2048 ? 2 : r3 < 65536 ? 3 : 4;
          for (t3 = new n2.Buf8(c2), o2 = 0, i2 = 0; o2 < c2; i2++)
            55296 == (64512 & (r3 = e3.charCodeAt(i2))) && i2 + 1 < a2 && 56320 == (64512 & (s2 = e3.charCodeAt(i2 + 1))) && (r3 = 65536 + (r3 - 55296 << 10) + (s2 - 56320), i2++), r3 < 128 ? t3[o2++] = r3 : r3 < 2048 ? (t3[o2++] = 192 | r3 >>> 6, t3[o2++] = 128 | 63 & r3) : r3 < 65536 ? (t3[o2++] = 224 | r3 >>> 12, t3[o2++] = 128 | r3 >>> 6 & 63, t3[o2++] = 128 | 63 & r3) : (t3[o2++] = 240 | r3 >>> 18, t3[o2++] = 128 | r3 >>> 12 & 63, t3[o2++] = 128 | r3 >>> 6 & 63, t3[o2++] = 128 | 63 & r3);
          return t3;
        }, t2.buf2binstring = function(e3) {
          return c(e3, e3.length);
        }, t2.binstring2buf = function(e3) {
          for (var t3 = new n2.Buf8(e3.length), r3 = 0, s2 = t3.length; r3 < s2; r3++)
            t3[r3] = e3.charCodeAt(r3);
          return t3;
        }, t2.buf2string = function(e3, t3) {
          var r3, n3, s2, i2, a2 = t3 || e3.length, u = new Array(2 * a2);
          for (n3 = 0, r3 = 0; r3 < a2; )
            if ((s2 = e3[r3++]) < 128)
              u[n3++] = s2;
            else if ((i2 = o[s2]) > 4)
              u[n3++] = 65533, r3 += i2 - 1;
            else {
              for (s2 &= 2 === i2 ? 31 : 3 === i2 ? 15 : 7; i2 > 1 && r3 < a2; )
                s2 = s2 << 6 | 63 & e3[r3++], i2--;
              i2 > 1 ? u[n3++] = 65533 : s2 < 65536 ? u[n3++] = s2 : (s2 -= 65536, u[n3++] = 55296 | s2 >> 10 & 1023, u[n3++] = 56320 | 1023 & s2);
            }
          return c(u, n3);
        }, t2.utf8border = function(e3, t3) {
          var r3;
          for ((t3 = t3 || e3.length) > e3.length && (t3 = e3.length), r3 = t3 - 1; r3 >= 0 && 128 == (192 & e3[r3]); )
            r3--;
          return r3 < 0 || 0 === r3 ? t3 : r3 + o[e3[r3]] > t3 ? r3 : t3;
        };
      }, 6069: (e2) => {
        "use strict";
        e2.exports = function(e3, t2, r2, n2) {
          for (var s = 65535 & e3 | 0, i = e3 >>> 16 & 65535 | 0, o = 0; 0 !== r2; ) {
            r2 -= o = r2 > 2e3 ? 2e3 : r2;
            do {
              i = i + (s = s + t2[n2++] | 0) | 0;
            } while (--o);
            s %= 65521, i %= 65521;
          }
          return s | i << 16 | 0;
        };
      }, 1619: (e2) => {
        "use strict";
        e2.exports = { Z_NO_FLUSH: 0, Z_PARTIAL_FLUSH: 1, Z_SYNC_FLUSH: 2, Z_FULL_FLUSH: 3, Z_FINISH: 4, Z_BLOCK: 5, Z_TREES: 6, Z_OK: 0, Z_STREAM_END: 1, Z_NEED_DICT: 2, Z_ERRNO: -1, Z_STREAM_ERROR: -2, Z_DATA_ERROR: -3, Z_BUF_ERROR: -5, Z_NO_COMPRESSION: 0, Z_BEST_SPEED: 1, Z_BEST_COMPRESSION: 9, Z_DEFAULT_COMPRESSION: -1, Z_FILTERED: 1, Z_HUFFMAN_ONLY: 2, Z_RLE: 3, Z_FIXED: 4, Z_DEFAULT_STRATEGY: 0, Z_BINARY: 0, Z_TEXT: 1, Z_UNKNOWN: 2, Z_DEFLATED: 8 };
      }, 2869: (e2) => {
        "use strict";
        var t2 = function() {
          for (var e3, t3 = [], r2 = 0; r2 < 256; r2++) {
            e3 = r2;
            for (var n2 = 0; n2 < 8; n2++)
              e3 = 1 & e3 ? 3988292384 ^ e3 >>> 1 : e3 >>> 1;
            t3[r2] = e3;
          }
          return t3;
        }();
        e2.exports = function(e3, r2, n2, s) {
          var i = t2, o = s + n2;
          e3 ^= -1;
          for (var a = s; a < o; a++)
            e3 = e3 >>> 8 ^ i[255 & (e3 ^ r2[a])];
          return -1 ^ e3;
        };
      }, 405: (e2, t2, r2) => {
        "use strict";
        var n2, s = r2(4236), i = r2(342), o = r2(6069), a = r2(2869), c = r2(8898), u = -2, l = 258, h = 262, p = 103, d = 113, f = 666;
        function m(e3, t3) {
          return e3.msg = c[t3], t3;
        }
        function w(e3) {
          return (e3 << 1) - (e3 > 4 ? 9 : 0);
        }
        function g(e3) {
          for (var t3 = e3.length; --t3 >= 0; )
            e3[t3] = 0;
        }
        function y(e3) {
          var t3 = e3.state, r3 = t3.pending;
          r3 > e3.avail_out && (r3 = e3.avail_out), 0 !== r3 && (s.arraySet(e3.output, t3.pending_buf, t3.pending_out, r3, e3.next_out), e3.next_out += r3, t3.pending_out += r3, e3.total_out += r3, e3.avail_out -= r3, t3.pending -= r3, 0 === t3.pending && (t3.pending_out = 0));
        }
        function b(e3, t3) {
          i._tr_flush_block(e3, e3.block_start >= 0 ? e3.block_start : -1, e3.strstart - e3.block_start, t3), e3.block_start = e3.strstart, y(e3.strm);
        }
        function x(e3, t3) {
          e3.pending_buf[e3.pending++] = t3;
        }
        function v(e3, t3) {
          e3.pending_buf[e3.pending++] = t3 >>> 8 & 255, e3.pending_buf[e3.pending++] = 255 & t3;
        }
        function _(e3, t3) {
          var r3, n3, s2 = e3.max_chain_length, i2 = e3.strstart, o2 = e3.prev_length, a2 = e3.nice_match, c2 = e3.strstart > e3.w_size - h ? e3.strstart - (e3.w_size - h) : 0, u2 = e3.window, p2 = e3.w_mask, d2 = e3.prev, f2 = e3.strstart + l, m2 = u2[i2 + o2 - 1], w2 = u2[i2 + o2];
          e3.prev_length >= e3.good_match && (s2 >>= 2), a2 > e3.lookahead && (a2 = e3.lookahead);
          do {
            if (u2[(r3 = t3) + o2] === w2 && u2[r3 + o2 - 1] === m2 && u2[r3] === u2[i2] && u2[++r3] === u2[i2 + 1]) {
              i2 += 2, r3++;
              do {
              } while (u2[++i2] === u2[++r3] && u2[++i2] === u2[++r3] && u2[++i2] === u2[++r3] && u2[++i2] === u2[++r3] && u2[++i2] === u2[++r3] && u2[++i2] === u2[++r3] && u2[++i2] === u2[++r3] && u2[++i2] === u2[++r3] && i2 < f2);
              if (n3 = l - (f2 - i2), i2 = f2 - l, n3 > o2) {
                if (e3.match_start = t3, o2 = n3, n3 >= a2)
                  break;
                m2 = u2[i2 + o2 - 1], w2 = u2[i2 + o2];
              }
            }
          } while ((t3 = d2[t3 & p2]) > c2 && 0 != --s2);
          return o2 <= e3.lookahead ? o2 : e3.lookahead;
        }
        function E(e3) {
          var t3, r3, n3, i2, c2, u2, l2, p2, d2, f2, m2 = e3.w_size;
          do {
            if (i2 = e3.window_size - e3.lookahead - e3.strstart, e3.strstart >= m2 + (m2 - h)) {
              s.arraySet(e3.window, e3.window, m2, m2, 0), e3.match_start -= m2, e3.strstart -= m2, e3.block_start -= m2, t3 = r3 = e3.hash_size;
              do {
                n3 = e3.head[--t3], e3.head[t3] = n3 >= m2 ? n3 - m2 : 0;
              } while (--r3);
              t3 = r3 = m2;
              do {
                n3 = e3.prev[--t3], e3.prev[t3] = n3 >= m2 ? n3 - m2 : 0;
              } while (--r3);
              i2 += m2;
            }
            if (0 === e3.strm.avail_in)
              break;
            if (u2 = e3.strm, l2 = e3.window, p2 = e3.strstart + e3.lookahead, d2 = i2, f2 = void 0, (f2 = u2.avail_in) > d2 && (f2 = d2), r3 = 0 === f2 ? 0 : (u2.avail_in -= f2, s.arraySet(l2, u2.input, u2.next_in, f2, p2), 1 === u2.state.wrap ? u2.adler = o(u2.adler, l2, f2, p2) : 2 === u2.state.wrap && (u2.adler = a(u2.adler, l2, f2, p2)), u2.next_in += f2, u2.total_in += f2, f2), e3.lookahead += r3, e3.lookahead + e3.insert >= 3)
              for (c2 = e3.strstart - e3.insert, e3.ins_h = e3.window[c2], e3.ins_h = (e3.ins_h << e3.hash_shift ^ e3.window[c2 + 1]) & e3.hash_mask; e3.insert && (e3.ins_h = (e3.ins_h << e3.hash_shift ^ e3.window[c2 + 3 - 1]) & e3.hash_mask, e3.prev[c2 & e3.w_mask] = e3.head[e3.ins_h], e3.head[e3.ins_h] = c2, c2++, e3.insert--, !(e3.lookahead + e3.insert < 3)); )
                ;
          } while (e3.lookahead < h && 0 !== e3.strm.avail_in);
        }
        function T(e3, t3) {
          for (var r3, n3; ; ) {
            if (e3.lookahead < h) {
              if (E(e3), e3.lookahead < h && 0 === t3)
                return 1;
              if (0 === e3.lookahead)
                break;
            }
            if (r3 = 0, e3.lookahead >= 3 && (e3.ins_h = (e3.ins_h << e3.hash_shift ^ e3.window[e3.strstart + 3 - 1]) & e3.hash_mask, r3 = e3.prev[e3.strstart & e3.w_mask] = e3.head[e3.ins_h], e3.head[e3.ins_h] = e3.strstart), 0 !== r3 && e3.strstart - r3 <= e3.w_size - h && (e3.match_length = _(e3, r3)), e3.match_length >= 3)
              if (n3 = i._tr_tally(e3, e3.strstart - e3.match_start, e3.match_length - 3), e3.lookahead -= e3.match_length, e3.match_length <= e3.max_lazy_match && e3.lookahead >= 3) {
                e3.match_length--;
                do {
                  e3.strstart++, e3.ins_h = (e3.ins_h << e3.hash_shift ^ e3.window[e3.strstart + 3 - 1]) & e3.hash_mask, r3 = e3.prev[e3.strstart & e3.w_mask] = e3.head[e3.ins_h], e3.head[e3.ins_h] = e3.strstart;
                } while (0 != --e3.match_length);
                e3.strstart++;
              } else
                e3.strstart += e3.match_length, e3.match_length = 0, e3.ins_h = e3.window[e3.strstart], e3.ins_h = (e3.ins_h << e3.hash_shift ^ e3.window[e3.strstart + 1]) & e3.hash_mask;
            else
              n3 = i._tr_tally(e3, 0, e3.window[e3.strstart]), e3.lookahead--, e3.strstart++;
            if (n3 && (b(e3, false), 0 === e3.strm.avail_out))
              return 1;
          }
          return e3.insert = e3.strstart < 2 ? e3.strstart : 2, 4 === t3 ? (b(e3, true), 0 === e3.strm.avail_out ? 3 : 4) : e3.last_lit && (b(e3, false), 0 === e3.strm.avail_out) ? 1 : 2;
        }
        function A(e3, t3) {
          for (var r3, n3, s2; ; ) {
            if (e3.lookahead < h) {
              if (E(e3), e3.lookahead < h && 0 === t3)
                return 1;
              if (0 === e3.lookahead)
                break;
            }
            if (r3 = 0, e3.lookahead >= 3 && (e3.ins_h = (e3.ins_h << e3.hash_shift ^ e3.window[e3.strstart + 3 - 1]) & e3.hash_mask, r3 = e3.prev[e3.strstart & e3.w_mask] = e3.head[e3.ins_h], e3.head[e3.ins_h] = e3.strstart), e3.prev_length = e3.match_length, e3.prev_match = e3.match_start, e3.match_length = 2, 0 !== r3 && e3.prev_length < e3.max_lazy_match && e3.strstart - r3 <= e3.w_size - h && (e3.match_length = _(e3, r3), e3.match_length <= 5 && (1 === e3.strategy || 3 === e3.match_length && e3.strstart - e3.match_start > 4096) && (e3.match_length = 2)), e3.prev_length >= 3 && e3.match_length <= e3.prev_length) {
              s2 = e3.strstart + e3.lookahead - 3, n3 = i._tr_tally(e3, e3.strstart - 1 - e3.prev_match, e3.prev_length - 3), e3.lookahead -= e3.prev_length - 1, e3.prev_length -= 2;
              do {
                ++e3.strstart <= s2 && (e3.ins_h = (e3.ins_h << e3.hash_shift ^ e3.window[e3.strstart + 3 - 1]) & e3.hash_mask, r3 = e3.prev[e3.strstart & e3.w_mask] = e3.head[e3.ins_h], e3.head[e3.ins_h] = e3.strstart);
              } while (0 != --e3.prev_length);
              if (e3.match_available = 0, e3.match_length = 2, e3.strstart++, n3 && (b(e3, false), 0 === e3.strm.avail_out))
                return 1;
            } else if (e3.match_available) {
              if ((n3 = i._tr_tally(e3, 0, e3.window[e3.strstart - 1])) && b(e3, false), e3.strstart++, e3.lookahead--, 0 === e3.strm.avail_out)
                return 1;
            } else
              e3.match_available = 1, e3.strstart++, e3.lookahead--;
          }
          return e3.match_available && (n3 = i._tr_tally(e3, 0, e3.window[e3.strstart - 1]), e3.match_available = 0), e3.insert = e3.strstart < 2 ? e3.strstart : 2, 4 === t3 ? (b(e3, true), 0 === e3.strm.avail_out ? 3 : 4) : e3.last_lit && (b(e3, false), 0 === e3.strm.avail_out) ? 1 : 2;
        }
        function S(e3, t3, r3, n3, s2) {
          this.good_length = e3, this.max_lazy = t3, this.nice_length = r3, this.max_chain = n3, this.func = s2;
        }
        function I() {
          this.strm = null, this.status = 0, this.pending_buf = null, this.pending_buf_size = 0, this.pending_out = 0, this.pending = 0, this.wrap = 0, this.gzhead = null, this.gzindex = 0, this.method = 8, this.last_flush = -1, this.w_size = 0, this.w_bits = 0, this.w_mask = 0, this.window = null, this.window_size = 0, this.prev = null, this.head = null, this.ins_h = 0, this.hash_size = 0, this.hash_bits = 0, this.hash_mask = 0, this.hash_shift = 0, this.block_start = 0, this.match_length = 0, this.prev_match = 0, this.match_available = 0, this.strstart = 0, this.match_start = 0, this.lookahead = 0, this.prev_length = 0, this.max_chain_length = 0, this.max_lazy_match = 0, this.level = 0, this.strategy = 0, this.good_match = 0, this.nice_match = 0, this.dyn_ltree = new s.Buf16(1146), this.dyn_dtree = new s.Buf16(122), this.bl_tree = new s.Buf16(78), g(this.dyn_ltree), g(this.dyn_dtree), g(this.bl_tree), this.l_desc = null, this.d_desc = null, this.bl_desc = null, this.bl_count = new s.Buf16(16), this.heap = new s.Buf16(573), g(this.heap), this.heap_len = 0, this.heap_max = 0, this.depth = new s.Buf16(573), g(this.depth), this.l_buf = 0, this.lit_bufsize = 0, this.last_lit = 0, this.d_buf = 0, this.opt_len = 0, this.static_len = 0, this.matches = 0, this.insert = 0, this.bi_buf = 0, this.bi_valid = 0;
        }
        function R(e3) {
          var t3;
          return e3 && e3.state ? (e3.total_in = e3.total_out = 0, e3.data_type = 2, (t3 = e3.state).pending = 0, t3.pending_out = 0, t3.wrap < 0 && (t3.wrap = -t3.wrap), t3.status = t3.wrap ? 42 : d, e3.adler = 2 === t3.wrap ? 0 : 1, t3.last_flush = 0, i._tr_init(t3), 0) : m(e3, u);
        }
        function N(e3) {
          var t3, r3 = R(e3);
          return 0 === r3 && ((t3 = e3.state).window_size = 2 * t3.w_size, g(t3.head), t3.max_lazy_match = n2[t3.level].max_lazy, t3.good_match = n2[t3.level].good_length, t3.nice_match = n2[t3.level].nice_length, t3.max_chain_length = n2[t3.level].max_chain, t3.strstart = 0, t3.block_start = 0, t3.lookahead = 0, t3.insert = 0, t3.match_length = t3.prev_length = 2, t3.match_available = 0, t3.ins_h = 0), r3;
        }
        function C(e3, t3, r3, n3, i2, o2) {
          if (!e3)
            return u;
          var a2 = 1;
          if (-1 === t3 && (t3 = 6), n3 < 0 ? (a2 = 0, n3 = -n3) : n3 > 15 && (a2 = 2, n3 -= 16), i2 < 1 || i2 > 9 || 8 !== r3 || n3 < 8 || n3 > 15 || t3 < 0 || t3 > 9 || o2 < 0 || o2 > 4)
            return m(e3, u);
          8 === n3 && (n3 = 9);
          var c2 = new I();
          return e3.state = c2, c2.strm = e3, c2.wrap = a2, c2.gzhead = null, c2.w_bits = n3, c2.w_size = 1 << c2.w_bits, c2.w_mask = c2.w_size - 1, c2.hash_bits = i2 + 7, c2.hash_size = 1 << c2.hash_bits, c2.hash_mask = c2.hash_size - 1, c2.hash_shift = ~~((c2.hash_bits + 3 - 1) / 3), c2.window = new s.Buf8(2 * c2.w_size), c2.head = new s.Buf16(c2.hash_size), c2.prev = new s.Buf16(c2.w_size), c2.lit_bufsize = 1 << i2 + 6, c2.pending_buf_size = 4 * c2.lit_bufsize, c2.pending_buf = new s.Buf8(c2.pending_buf_size), c2.d_buf = 1 * c2.lit_bufsize, c2.l_buf = 3 * c2.lit_bufsize, c2.level = t3, c2.strategy = o2, c2.method = r3, N(e3);
        }
        n2 = [new S(0, 0, 0, 0, function(e3, t3) {
          var r3 = 65535;
          for (r3 > e3.pending_buf_size - 5 && (r3 = e3.pending_buf_size - 5); ; ) {
            if (e3.lookahead <= 1) {
              if (E(e3), 0 === e3.lookahead && 0 === t3)
                return 1;
              if (0 === e3.lookahead)
                break;
            }
            e3.strstart += e3.lookahead, e3.lookahead = 0;
            var n3 = e3.block_start + r3;
            if ((0 === e3.strstart || e3.strstart >= n3) && (e3.lookahead = e3.strstart - n3, e3.strstart = n3, b(e3, false), 0 === e3.strm.avail_out))
              return 1;
            if (e3.strstart - e3.block_start >= e3.w_size - h && (b(e3, false), 0 === e3.strm.avail_out))
              return 1;
          }
          return e3.insert = 0, 4 === t3 ? (b(e3, true), 0 === e3.strm.avail_out ? 3 : 4) : (e3.strstart > e3.block_start && (b(e3, false), e3.strm.avail_out), 1);
        }), new S(4, 4, 8, 4, T), new S(4, 5, 16, 8, T), new S(4, 6, 32, 32, T), new S(4, 4, 16, 16, A), new S(8, 16, 32, 32, A), new S(8, 16, 128, 128, A), new S(8, 32, 128, 256, A), new S(32, 128, 258, 1024, A), new S(32, 258, 258, 4096, A)], t2.deflateInit = function(e3, t3) {
          return C(e3, t3, 8, 15, 8, 0);
        }, t2.deflateInit2 = C, t2.deflateReset = N, t2.deflateResetKeep = R, t2.deflateSetHeader = function(e3, t3) {
          return e3 && e3.state ? 2 !== e3.state.wrap ? u : (e3.state.gzhead = t3, 0) : u;
        }, t2.deflate = function(e3, t3) {
          var r3, s2, o2, c2;
          if (!e3 || !e3.state || t3 > 5 || t3 < 0)
            return e3 ? m(e3, u) : u;
          if (s2 = e3.state, !e3.output || !e3.input && 0 !== e3.avail_in || s2.status === f && 4 !== t3)
            return m(e3, 0 === e3.avail_out ? -5 : u);
          if (s2.strm = e3, r3 = s2.last_flush, s2.last_flush = t3, 42 === s2.status)
            if (2 === s2.wrap)
              e3.adler = 0, x(s2, 31), x(s2, 139), x(s2, 8), s2.gzhead ? (x(s2, (s2.gzhead.text ? 1 : 0) + (s2.gzhead.hcrc ? 2 : 0) + (s2.gzhead.extra ? 4 : 0) + (s2.gzhead.name ? 8 : 0) + (s2.gzhead.comment ? 16 : 0)), x(s2, 255 & s2.gzhead.time), x(s2, s2.gzhead.time >> 8 & 255), x(s2, s2.gzhead.time >> 16 & 255), x(s2, s2.gzhead.time >> 24 & 255), x(s2, 9 === s2.level ? 2 : s2.strategy >= 2 || s2.level < 2 ? 4 : 0), x(s2, 255 & s2.gzhead.os), s2.gzhead.extra && s2.gzhead.extra.length && (x(s2, 255 & s2.gzhead.extra.length), x(s2, s2.gzhead.extra.length >> 8 & 255)), s2.gzhead.hcrc && (e3.adler = a(e3.adler, s2.pending_buf, s2.pending, 0)), s2.gzindex = 0, s2.status = 69) : (x(s2, 0), x(s2, 0), x(s2, 0), x(s2, 0), x(s2, 0), x(s2, 9 === s2.level ? 2 : s2.strategy >= 2 || s2.level < 2 ? 4 : 0), x(s2, 3), s2.status = d);
            else {
              var h2 = 8 + (s2.w_bits - 8 << 4) << 8;
              h2 |= (s2.strategy >= 2 || s2.level < 2 ? 0 : s2.level < 6 ? 1 : 6 === s2.level ? 2 : 3) << 6, 0 !== s2.strstart && (h2 |= 32), h2 += 31 - h2 % 31, s2.status = d, v(s2, h2), 0 !== s2.strstart && (v(s2, e3.adler >>> 16), v(s2, 65535 & e3.adler)), e3.adler = 1;
            }
          if (69 === s2.status)
            if (s2.gzhead.extra) {
              for (o2 = s2.pending; s2.gzindex < (65535 & s2.gzhead.extra.length) && (s2.pending !== s2.pending_buf_size || (s2.gzhead.hcrc && s2.pending > o2 && (e3.adler = a(e3.adler, s2.pending_buf, s2.pending - o2, o2)), y(e3), o2 = s2.pending, s2.pending !== s2.pending_buf_size)); )
                x(s2, 255 & s2.gzhead.extra[s2.gzindex]), s2.gzindex++;
              s2.gzhead.hcrc && s2.pending > o2 && (e3.adler = a(e3.adler, s2.pending_buf, s2.pending - o2, o2)), s2.gzindex === s2.gzhead.extra.length && (s2.gzindex = 0, s2.status = 73);
            } else
              s2.status = 73;
          if (73 === s2.status)
            if (s2.gzhead.name) {
              o2 = s2.pending;
              do {
                if (s2.pending === s2.pending_buf_size && (s2.gzhead.hcrc && s2.pending > o2 && (e3.adler = a(e3.adler, s2.pending_buf, s2.pending - o2, o2)), y(e3), o2 = s2.pending, s2.pending === s2.pending_buf_size)) {
                  c2 = 1;
                  break;
                }
                c2 = s2.gzindex < s2.gzhead.name.length ? 255 & s2.gzhead.name.charCodeAt(s2.gzindex++) : 0, x(s2, c2);
              } while (0 !== c2);
              s2.gzhead.hcrc && s2.pending > o2 && (e3.adler = a(e3.adler, s2.pending_buf, s2.pending - o2, o2)), 0 === c2 && (s2.gzindex = 0, s2.status = 91);
            } else
              s2.status = 91;
          if (91 === s2.status)
            if (s2.gzhead.comment) {
              o2 = s2.pending;
              do {
                if (s2.pending === s2.pending_buf_size && (s2.gzhead.hcrc && s2.pending > o2 && (e3.adler = a(e3.adler, s2.pending_buf, s2.pending - o2, o2)), y(e3), o2 = s2.pending, s2.pending === s2.pending_buf_size)) {
                  c2 = 1;
                  break;
                }
                c2 = s2.gzindex < s2.gzhead.comment.length ? 255 & s2.gzhead.comment.charCodeAt(s2.gzindex++) : 0, x(s2, c2);
              } while (0 !== c2);
              s2.gzhead.hcrc && s2.pending > o2 && (e3.adler = a(e3.adler, s2.pending_buf, s2.pending - o2, o2)), 0 === c2 && (s2.status = p);
            } else
              s2.status = p;
          if (s2.status === p && (s2.gzhead.hcrc ? (s2.pending + 2 > s2.pending_buf_size && y(e3), s2.pending + 2 <= s2.pending_buf_size && (x(s2, 255 & e3.adler), x(s2, e3.adler >> 8 & 255), e3.adler = 0, s2.status = d)) : s2.status = d), 0 !== s2.pending) {
            if (y(e3), 0 === e3.avail_out)
              return s2.last_flush = -1, 0;
          } else if (0 === e3.avail_in && w(t3) <= w(r3) && 4 !== t3)
            return m(e3, -5);
          if (s2.status === f && 0 !== e3.avail_in)
            return m(e3, -5);
          if (0 !== e3.avail_in || 0 !== s2.lookahead || 0 !== t3 && s2.status !== f) {
            var _2 = 2 === s2.strategy ? function(e4, t4) {
              for (var r4; ; ) {
                if (0 === e4.lookahead && (E(e4), 0 === e4.lookahead)) {
                  if (0 === t4)
                    return 1;
                  break;
                }
                if (e4.match_length = 0, r4 = i._tr_tally(e4, 0, e4.window[e4.strstart]), e4.lookahead--, e4.strstart++, r4 && (b(e4, false), 0 === e4.strm.avail_out))
                  return 1;
              }
              return e4.insert = 0, 4 === t4 ? (b(e4, true), 0 === e4.strm.avail_out ? 3 : 4) : e4.last_lit && (b(e4, false), 0 === e4.strm.avail_out) ? 1 : 2;
            }(s2, t3) : 3 === s2.strategy ? function(e4, t4) {
              for (var r4, n3, s3, o3, a2 = e4.window; ; ) {
                if (e4.lookahead <= l) {
                  if (E(e4), e4.lookahead <= l && 0 === t4)
                    return 1;
                  if (0 === e4.lookahead)
                    break;
                }
                if (e4.match_length = 0, e4.lookahead >= 3 && e4.strstart > 0 && (n3 = a2[s3 = e4.strstart - 1]) === a2[++s3] && n3 === a2[++s3] && n3 === a2[++s3]) {
                  o3 = e4.strstart + l;
                  do {
                  } while (n3 === a2[++s3] && n3 === a2[++s3] && n3 === a2[++s3] && n3 === a2[++s3] && n3 === a2[++s3] && n3 === a2[++s3] && n3 === a2[++s3] && n3 === a2[++s3] && s3 < o3);
                  e4.match_length = l - (o3 - s3), e4.match_length > e4.lookahead && (e4.match_length = e4.lookahead);
                }
                if (e4.match_length >= 3 ? (r4 = i._tr_tally(e4, 1, e4.match_length - 3), e4.lookahead -= e4.match_length, e4.strstart += e4.match_length, e4.match_length = 0) : (r4 = i._tr_tally(e4, 0, e4.window[e4.strstart]), e4.lookahead--, e4.strstart++), r4 && (b(e4, false), 0 === e4.strm.avail_out))
                  return 1;
              }
              return e4.insert = 0, 4 === t4 ? (b(e4, true), 0 === e4.strm.avail_out ? 3 : 4) : e4.last_lit && (b(e4, false), 0 === e4.strm.avail_out) ? 1 : 2;
            }(s2, t3) : n2[s2.level].func(s2, t3);
            if (3 !== _2 && 4 !== _2 || (s2.status = f), 1 === _2 || 3 === _2)
              return 0 === e3.avail_out && (s2.last_flush = -1), 0;
            if (2 === _2 && (1 === t3 ? i._tr_align(s2) : 5 !== t3 && (i._tr_stored_block(s2, 0, 0, false), 3 === t3 && (g(s2.head), 0 === s2.lookahead && (s2.strstart = 0, s2.block_start = 0, s2.insert = 0))), y(e3), 0 === e3.avail_out))
              return s2.last_flush = -1, 0;
          }
          return 4 !== t3 ? 0 : s2.wrap <= 0 ? 1 : (2 === s2.wrap ? (x(s2, 255 & e3.adler), x(s2, e3.adler >> 8 & 255), x(s2, e3.adler >> 16 & 255), x(s2, e3.adler >> 24 & 255), x(s2, 255 & e3.total_in), x(s2, e3.total_in >> 8 & 255), x(s2, e3.total_in >> 16 & 255), x(s2, e3.total_in >> 24 & 255)) : (v(s2, e3.adler >>> 16), v(s2, 65535 & e3.adler)), y(e3), s2.wrap > 0 && (s2.wrap = -s2.wrap), 0 !== s2.pending ? 0 : 1);
        }, t2.deflateEnd = function(e3) {
          var t3;
          return e3 && e3.state ? 42 !== (t3 = e3.state.status) && 69 !== t3 && 73 !== t3 && 91 !== t3 && t3 !== p && t3 !== d && t3 !== f ? m(e3, u) : (e3.state = null, t3 === d ? m(e3, -3) : 0) : u;
        }, t2.deflateSetDictionary = function(e3, t3) {
          var r3, n3, i2, a2, c2, l2, h2, p2, d2 = t3.length;
          if (!e3 || !e3.state)
            return u;
          if (2 === (a2 = (r3 = e3.state).wrap) || 1 === a2 && 42 !== r3.status || r3.lookahead)
            return u;
          for (1 === a2 && (e3.adler = o(e3.adler, t3, d2, 0)), r3.wrap = 0, d2 >= r3.w_size && (0 === a2 && (g(r3.head), r3.strstart = 0, r3.block_start = 0, r3.insert = 0), p2 = new s.Buf8(r3.w_size), s.arraySet(p2, t3, d2 - r3.w_size, r3.w_size, 0), t3 = p2, d2 = r3.w_size), c2 = e3.avail_in, l2 = e3.next_in, h2 = e3.input, e3.avail_in = d2, e3.next_in = 0, e3.input = t3, E(r3); r3.lookahead >= 3; ) {
            n3 = r3.strstart, i2 = r3.lookahead - 2;
            do {
              r3.ins_h = (r3.ins_h << r3.hash_shift ^ r3.window[n3 + 3 - 1]) & r3.hash_mask, r3.prev[n3 & r3.w_mask] = r3.head[r3.ins_h], r3.head[r3.ins_h] = n3, n3++;
            } while (--i2);
            r3.strstart = n3, r3.lookahead = 2, E(r3);
          }
          return r3.strstart += r3.lookahead, r3.block_start = r3.strstart, r3.insert = r3.lookahead, r3.lookahead = 0, r3.match_length = r3.prev_length = 2, r3.match_available = 0, e3.next_in = l2, e3.input = h2, e3.avail_in = c2, r3.wrap = a2, 0;
        }, t2.deflateInfo = "pako deflate (from Nodeca project)";
      }, 2401: (e2) => {
        "use strict";
        e2.exports = function() {
          this.text = 0, this.time = 0, this.xflags = 0, this.os = 0, this.extra = null, this.extra_len = 0, this.name = "", this.comment = "", this.hcrc = 0, this.done = false;
        };
      }, 4264: (e2) => {
        "use strict";
        e2.exports = function(e3, t2) {
          var r2, n2, s, i, o, a, c, u, l, h, p, d, f, m, w, g, y, b, x, v, _, E, T, A, S;
          r2 = e3.state, n2 = e3.next_in, A = e3.input, s = n2 + (e3.avail_in - 5), i = e3.next_out, S = e3.output, o = i - (t2 - e3.avail_out), a = i + (e3.avail_out - 257), c = r2.dmax, u = r2.wsize, l = r2.whave, h = r2.wnext, p = r2.window, d = r2.hold, f = r2.bits, m = r2.lencode, w = r2.distcode, g = (1 << r2.lenbits) - 1, y = (1 << r2.distbits) - 1;
          e:
            do {
              f < 15 && (d += A[n2++] << f, f += 8, d += A[n2++] << f, f += 8), b = m[d & g];
              t:
                for (; ; ) {
                  if (d >>>= x = b >>> 24, f -= x, 0 == (x = b >>> 16 & 255))
                    S[i++] = 65535 & b;
                  else {
                    if (!(16 & x)) {
                      if (0 == (64 & x)) {
                        b = m[(65535 & b) + (d & (1 << x) - 1)];
                        continue t;
                      }
                      if (32 & x) {
                        r2.mode = 12;
                        break e;
                      }
                      e3.msg = "invalid literal/length code", r2.mode = 30;
                      break e;
                    }
                    v = 65535 & b, (x &= 15) && (f < x && (d += A[n2++] << f, f += 8), v += d & (1 << x) - 1, d >>>= x, f -= x), f < 15 && (d += A[n2++] << f, f += 8, d += A[n2++] << f, f += 8), b = w[d & y];
                    r:
                      for (; ; ) {
                        if (d >>>= x = b >>> 24, f -= x, !(16 & (x = b >>> 16 & 255))) {
                          if (0 == (64 & x)) {
                            b = w[(65535 & b) + (d & (1 << x) - 1)];
                            continue r;
                          }
                          e3.msg = "invalid distance code", r2.mode = 30;
                          break e;
                        }
                        if (_ = 65535 & b, f < (x &= 15) && (d += A[n2++] << f, (f += 8) < x && (d += A[n2++] << f, f += 8)), (_ += d & (1 << x) - 1) > c) {
                          e3.msg = "invalid distance too far back", r2.mode = 30;
                          break e;
                        }
                        if (d >>>= x, f -= x, _ > (x = i - o)) {
                          if ((x = _ - x) > l && r2.sane) {
                            e3.msg = "invalid distance too far back", r2.mode = 30;
                            break e;
                          }
                          if (E = 0, T = p, 0 === h) {
                            if (E += u - x, x < v) {
                              v -= x;
                              do {
                                S[i++] = p[E++];
                              } while (--x);
                              E = i - _, T = S;
                            }
                          } else if (h < x) {
                            if (E += u + h - x, (x -= h) < v) {
                              v -= x;
                              do {
                                S[i++] = p[E++];
                              } while (--x);
                              if (E = 0, h < v) {
                                v -= x = h;
                                do {
                                  S[i++] = p[E++];
                                } while (--x);
                                E = i - _, T = S;
                              }
                            }
                          } else if (E += h - x, x < v) {
                            v -= x;
                            do {
                              S[i++] = p[E++];
                            } while (--x);
                            E = i - _, T = S;
                          }
                          for (; v > 2; )
                            S[i++] = T[E++], S[i++] = T[E++], S[i++] = T[E++], v -= 3;
                          v && (S[i++] = T[E++], v > 1 && (S[i++] = T[E++]));
                        } else {
                          E = i - _;
                          do {
                            S[i++] = S[E++], S[i++] = S[E++], S[i++] = S[E++], v -= 3;
                          } while (v > 2);
                          v && (S[i++] = S[E++], v > 1 && (S[i++] = S[E++]));
                        }
                        break;
                      }
                  }
                  break;
                }
            } while (n2 < s && i < a);
          n2 -= v = f >> 3, d &= (1 << (f -= v << 3)) - 1, e3.next_in = n2, e3.next_out = i, e3.avail_in = n2 < s ? s - n2 + 5 : 5 - (n2 - s), e3.avail_out = i < a ? a - i + 257 : 257 - (i - a), r2.hold = d, r2.bits = f;
        };
      }, 7948: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(4236), s = r2(6069), i = r2(2869), o = r2(4264), a = r2(9241), c = -2, u = 12, l = 30;
        function h(e3) {
          return (e3 >>> 24 & 255) + (e3 >>> 8 & 65280) + ((65280 & e3) << 8) + ((255 & e3) << 24);
        }
        function p() {
          this.mode = 0, this.last = false, this.wrap = 0, this.havedict = false, this.flags = 0, this.dmax = 0, this.check = 0, this.total = 0, this.head = null, this.wbits = 0, this.wsize = 0, this.whave = 0, this.wnext = 0, this.window = null, this.hold = 0, this.bits = 0, this.length = 0, this.offset = 0, this.extra = 0, this.lencode = null, this.distcode = null, this.lenbits = 0, this.distbits = 0, this.ncode = 0, this.nlen = 0, this.ndist = 0, this.have = 0, this.next = null, this.lens = new n2.Buf16(320), this.work = new n2.Buf16(288), this.lendyn = null, this.distdyn = null, this.sane = 0, this.back = 0, this.was = 0;
        }
        function d(e3) {
          var t3;
          return e3 && e3.state ? (t3 = e3.state, e3.total_in = e3.total_out = t3.total = 0, e3.msg = "", t3.wrap && (e3.adler = 1 & t3.wrap), t3.mode = 1, t3.last = 0, t3.havedict = 0, t3.dmax = 32768, t3.head = null, t3.hold = 0, t3.bits = 0, t3.lencode = t3.lendyn = new n2.Buf32(852), t3.distcode = t3.distdyn = new n2.Buf32(592), t3.sane = 1, t3.back = -1, 0) : c;
        }
        function f(e3) {
          var t3;
          return e3 && e3.state ? ((t3 = e3.state).wsize = 0, t3.whave = 0, t3.wnext = 0, d(e3)) : c;
        }
        function m(e3, t3) {
          var r3, n3;
          return e3 && e3.state ? (n3 = e3.state, t3 < 0 ? (r3 = 0, t3 = -t3) : (r3 = 1 + (t3 >> 4), t3 < 48 && (t3 &= 15)), t3 && (t3 < 8 || t3 > 15) ? c : (null !== n3.window && n3.wbits !== t3 && (n3.window = null), n3.wrap = r3, n3.wbits = t3, f(e3))) : c;
        }
        function w(e3, t3) {
          var r3, n3;
          return e3 ? (n3 = new p(), e3.state = n3, n3.window = null, 0 !== (r3 = m(e3, t3)) && (e3.state = null), r3) : c;
        }
        var g, y, b = true;
        function x(e3) {
          if (b) {
            var t3;
            for (g = new n2.Buf32(512), y = new n2.Buf32(32), t3 = 0; t3 < 144; )
              e3.lens[t3++] = 8;
            for (; t3 < 256; )
              e3.lens[t3++] = 9;
            for (; t3 < 280; )
              e3.lens[t3++] = 7;
            for (; t3 < 288; )
              e3.lens[t3++] = 8;
            for (a(1, e3.lens, 0, 288, g, 0, e3.work, { bits: 9 }), t3 = 0; t3 < 32; )
              e3.lens[t3++] = 5;
            a(2, e3.lens, 0, 32, y, 0, e3.work, { bits: 5 }), b = false;
          }
          e3.lencode = g, e3.lenbits = 9, e3.distcode = y, e3.distbits = 5;
        }
        function v(e3, t3, r3, s2) {
          var i2, o2 = e3.state;
          return null === o2.window && (o2.wsize = 1 << o2.wbits, o2.wnext = 0, o2.whave = 0, o2.window = new n2.Buf8(o2.wsize)), s2 >= o2.wsize ? (n2.arraySet(o2.window, t3, r3 - o2.wsize, o2.wsize, 0), o2.wnext = 0, o2.whave = o2.wsize) : ((i2 = o2.wsize - o2.wnext) > s2 && (i2 = s2), n2.arraySet(o2.window, t3, r3 - s2, i2, o2.wnext), (s2 -= i2) ? (n2.arraySet(o2.window, t3, r3 - s2, s2, 0), o2.wnext = s2, o2.whave = o2.wsize) : (o2.wnext += i2, o2.wnext === o2.wsize && (o2.wnext = 0), o2.whave < o2.wsize && (o2.whave += i2))), 0;
        }
        t2.inflateReset = f, t2.inflateReset2 = m, t2.inflateResetKeep = d, t2.inflateInit = function(e3) {
          return w(e3, 15);
        }, t2.inflateInit2 = w, t2.inflate = function(e3, t3) {
          var r3, p2, d2, f2, m2, w2, g2, y2, b2, _, E, T, A, S, I, R, N, C, O, k, L, D, P, F, B = 0, M = new n2.Buf8(4), U = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
          if (!e3 || !e3.state || !e3.output || !e3.input && 0 !== e3.avail_in)
            return c;
          (r3 = e3.state).mode === u && (r3.mode = 13), m2 = e3.next_out, d2 = e3.output, g2 = e3.avail_out, f2 = e3.next_in, p2 = e3.input, w2 = e3.avail_in, y2 = r3.hold, b2 = r3.bits, _ = w2, E = g2, D = 0;
          e:
            for (; ; )
              switch (r3.mode) {
                case 1:
                  if (0 === r3.wrap) {
                    r3.mode = 13;
                    break;
                  }
                  for (; b2 < 16; ) {
                    if (0 === w2)
                      break e;
                    w2--, y2 += p2[f2++] << b2, b2 += 8;
                  }
                  if (2 & r3.wrap && 35615 === y2) {
                    r3.check = 0, M[0] = 255 & y2, M[1] = y2 >>> 8 & 255, r3.check = i(r3.check, M, 2, 0), y2 = 0, b2 = 0, r3.mode = 2;
                    break;
                  }
                  if (r3.flags = 0, r3.head && (r3.head.done = false), !(1 & r3.wrap) || (((255 & y2) << 8) + (y2 >> 8)) % 31) {
                    e3.msg = "incorrect header check", r3.mode = l;
                    break;
                  }
                  if (8 != (15 & y2)) {
                    e3.msg = "unknown compression method", r3.mode = l;
                    break;
                  }
                  if (b2 -= 4, L = 8 + (15 & (y2 >>>= 4)), 0 === r3.wbits)
                    r3.wbits = L;
                  else if (L > r3.wbits) {
                    e3.msg = "invalid window size", r3.mode = l;
                    break;
                  }
                  r3.dmax = 1 << L, e3.adler = r3.check = 1, r3.mode = 512 & y2 ? 10 : u, y2 = 0, b2 = 0;
                  break;
                case 2:
                  for (; b2 < 16; ) {
                    if (0 === w2)
                      break e;
                    w2--, y2 += p2[f2++] << b2, b2 += 8;
                  }
                  if (r3.flags = y2, 8 != (255 & r3.flags)) {
                    e3.msg = "unknown compression method", r3.mode = l;
                    break;
                  }
                  if (57344 & r3.flags) {
                    e3.msg = "unknown header flags set", r3.mode = l;
                    break;
                  }
                  r3.head && (r3.head.text = y2 >> 8 & 1), 512 & r3.flags && (M[0] = 255 & y2, M[1] = y2 >>> 8 & 255, r3.check = i(r3.check, M, 2, 0)), y2 = 0, b2 = 0, r3.mode = 3;
                case 3:
                  for (; b2 < 32; ) {
                    if (0 === w2)
                      break e;
                    w2--, y2 += p2[f2++] << b2, b2 += 8;
                  }
                  r3.head && (r3.head.time = y2), 512 & r3.flags && (M[0] = 255 & y2, M[1] = y2 >>> 8 & 255, M[2] = y2 >>> 16 & 255, M[3] = y2 >>> 24 & 255, r3.check = i(r3.check, M, 4, 0)), y2 = 0, b2 = 0, r3.mode = 4;
                case 4:
                  for (; b2 < 16; ) {
                    if (0 === w2)
                      break e;
                    w2--, y2 += p2[f2++] << b2, b2 += 8;
                  }
                  r3.head && (r3.head.xflags = 255 & y2, r3.head.os = y2 >> 8), 512 & r3.flags && (M[0] = 255 & y2, M[1] = y2 >>> 8 & 255, r3.check = i(r3.check, M, 2, 0)), y2 = 0, b2 = 0, r3.mode = 5;
                case 5:
                  if (1024 & r3.flags) {
                    for (; b2 < 16; ) {
                      if (0 === w2)
                        break e;
                      w2--, y2 += p2[f2++] << b2, b2 += 8;
                    }
                    r3.length = y2, r3.head && (r3.head.extra_len = y2), 512 & r3.flags && (M[0] = 255 & y2, M[1] = y2 >>> 8 & 255, r3.check = i(r3.check, M, 2, 0)), y2 = 0, b2 = 0;
                  } else
                    r3.head && (r3.head.extra = null);
                  r3.mode = 6;
                case 6:
                  if (1024 & r3.flags && ((T = r3.length) > w2 && (T = w2), T && (r3.head && (L = r3.head.extra_len - r3.length, r3.head.extra || (r3.head.extra = new Array(r3.head.extra_len)), n2.arraySet(r3.head.extra, p2, f2, T, L)), 512 & r3.flags && (r3.check = i(r3.check, p2, T, f2)), w2 -= T, f2 += T, r3.length -= T), r3.length))
                    break e;
                  r3.length = 0, r3.mode = 7;
                case 7:
                  if (2048 & r3.flags) {
                    if (0 === w2)
                      break e;
                    T = 0;
                    do {
                      L = p2[f2 + T++], r3.head && L && r3.length < 65536 && (r3.head.name += String.fromCharCode(L));
                    } while (L && T < w2);
                    if (512 & r3.flags && (r3.check = i(r3.check, p2, T, f2)), w2 -= T, f2 += T, L)
                      break e;
                  } else
                    r3.head && (r3.head.name = null);
                  r3.length = 0, r3.mode = 8;
                case 8:
                  if (4096 & r3.flags) {
                    if (0 === w2)
                      break e;
                    T = 0;
                    do {
                      L = p2[f2 + T++], r3.head && L && r3.length < 65536 && (r3.head.comment += String.fromCharCode(L));
                    } while (L && T < w2);
                    if (512 & r3.flags && (r3.check = i(r3.check, p2, T, f2)), w2 -= T, f2 += T, L)
                      break e;
                  } else
                    r3.head && (r3.head.comment = null);
                  r3.mode = 9;
                case 9:
                  if (512 & r3.flags) {
                    for (; b2 < 16; ) {
                      if (0 === w2)
                        break e;
                      w2--, y2 += p2[f2++] << b2, b2 += 8;
                    }
                    if (y2 !== (65535 & r3.check)) {
                      e3.msg = "header crc mismatch", r3.mode = l;
                      break;
                    }
                    y2 = 0, b2 = 0;
                  }
                  r3.head && (r3.head.hcrc = r3.flags >> 9 & 1, r3.head.done = true), e3.adler = r3.check = 0, r3.mode = u;
                  break;
                case 10:
                  for (; b2 < 32; ) {
                    if (0 === w2)
                      break e;
                    w2--, y2 += p2[f2++] << b2, b2 += 8;
                  }
                  e3.adler = r3.check = h(y2), y2 = 0, b2 = 0, r3.mode = 11;
                case 11:
                  if (0 === r3.havedict)
                    return e3.next_out = m2, e3.avail_out = g2, e3.next_in = f2, e3.avail_in = w2, r3.hold = y2, r3.bits = b2, 2;
                  e3.adler = r3.check = 1, r3.mode = u;
                case u:
                  if (5 === t3 || 6 === t3)
                    break e;
                case 13:
                  if (r3.last) {
                    y2 >>>= 7 & b2, b2 -= 7 & b2, r3.mode = 27;
                    break;
                  }
                  for (; b2 < 3; ) {
                    if (0 === w2)
                      break e;
                    w2--, y2 += p2[f2++] << b2, b2 += 8;
                  }
                  switch (r3.last = 1 & y2, b2 -= 1, 3 & (y2 >>>= 1)) {
                    case 0:
                      r3.mode = 14;
                      break;
                    case 1:
                      if (x(r3), r3.mode = 20, 6 === t3) {
                        y2 >>>= 2, b2 -= 2;
                        break e;
                      }
                      break;
                    case 2:
                      r3.mode = 17;
                      break;
                    case 3:
                      e3.msg = "invalid block type", r3.mode = l;
                  }
                  y2 >>>= 2, b2 -= 2;
                  break;
                case 14:
                  for (y2 >>>= 7 & b2, b2 -= 7 & b2; b2 < 32; ) {
                    if (0 === w2)
                      break e;
                    w2--, y2 += p2[f2++] << b2, b2 += 8;
                  }
                  if ((65535 & y2) != (y2 >>> 16 ^ 65535)) {
                    e3.msg = "invalid stored block lengths", r3.mode = l;
                    break;
                  }
                  if (r3.length = 65535 & y2, y2 = 0, b2 = 0, r3.mode = 15, 6 === t3)
                    break e;
                case 15:
                  r3.mode = 16;
                case 16:
                  if (T = r3.length) {
                    if (T > w2 && (T = w2), T > g2 && (T = g2), 0 === T)
                      break e;
                    n2.arraySet(d2, p2, f2, T, m2), w2 -= T, f2 += T, g2 -= T, m2 += T, r3.length -= T;
                    break;
                  }
                  r3.mode = u;
                  break;
                case 17:
                  for (; b2 < 14; ) {
                    if (0 === w2)
                      break e;
                    w2--, y2 += p2[f2++] << b2, b2 += 8;
                  }
                  if (r3.nlen = 257 + (31 & y2), y2 >>>= 5, b2 -= 5, r3.ndist = 1 + (31 & y2), y2 >>>= 5, b2 -= 5, r3.ncode = 4 + (15 & y2), y2 >>>= 4, b2 -= 4, r3.nlen > 286 || r3.ndist > 30) {
                    e3.msg = "too many length or distance symbols", r3.mode = l;
                    break;
                  }
                  r3.have = 0, r3.mode = 18;
                case 18:
                  for (; r3.have < r3.ncode; ) {
                    for (; b2 < 3; ) {
                      if (0 === w2)
                        break e;
                      w2--, y2 += p2[f2++] << b2, b2 += 8;
                    }
                    r3.lens[U[r3.have++]] = 7 & y2, y2 >>>= 3, b2 -= 3;
                  }
                  for (; r3.have < 19; )
                    r3.lens[U[r3.have++]] = 0;
                  if (r3.lencode = r3.lendyn, r3.lenbits = 7, P = { bits: r3.lenbits }, D = a(0, r3.lens, 0, 19, r3.lencode, 0, r3.work, P), r3.lenbits = P.bits, D) {
                    e3.msg = "invalid code lengths set", r3.mode = l;
                    break;
                  }
                  r3.have = 0, r3.mode = 19;
                case 19:
                  for (; r3.have < r3.nlen + r3.ndist; ) {
                    for (; R = (B = r3.lencode[y2 & (1 << r3.lenbits) - 1]) >>> 16 & 255, N = 65535 & B, !((I = B >>> 24) <= b2); ) {
                      if (0 === w2)
                        break e;
                      w2--, y2 += p2[f2++] << b2, b2 += 8;
                    }
                    if (N < 16)
                      y2 >>>= I, b2 -= I, r3.lens[r3.have++] = N;
                    else {
                      if (16 === N) {
                        for (F = I + 2; b2 < F; ) {
                          if (0 === w2)
                            break e;
                          w2--, y2 += p2[f2++] << b2, b2 += 8;
                        }
                        if (y2 >>>= I, b2 -= I, 0 === r3.have) {
                          e3.msg = "invalid bit length repeat", r3.mode = l;
                          break;
                        }
                        L = r3.lens[r3.have - 1], T = 3 + (3 & y2), y2 >>>= 2, b2 -= 2;
                      } else if (17 === N) {
                        for (F = I + 3; b2 < F; ) {
                          if (0 === w2)
                            break e;
                          w2--, y2 += p2[f2++] << b2, b2 += 8;
                        }
                        b2 -= I, L = 0, T = 3 + (7 & (y2 >>>= I)), y2 >>>= 3, b2 -= 3;
                      } else {
                        for (F = I + 7; b2 < F; ) {
                          if (0 === w2)
                            break e;
                          w2--, y2 += p2[f2++] << b2, b2 += 8;
                        }
                        b2 -= I, L = 0, T = 11 + (127 & (y2 >>>= I)), y2 >>>= 7, b2 -= 7;
                      }
                      if (r3.have + T > r3.nlen + r3.ndist) {
                        e3.msg = "invalid bit length repeat", r3.mode = l;
                        break;
                      }
                      for (; T--; )
                        r3.lens[r3.have++] = L;
                    }
                  }
                  if (r3.mode === l)
                    break;
                  if (0 === r3.lens[256]) {
                    e3.msg = "invalid code -- missing end-of-block", r3.mode = l;
                    break;
                  }
                  if (r3.lenbits = 9, P = { bits: r3.lenbits }, D = a(1, r3.lens, 0, r3.nlen, r3.lencode, 0, r3.work, P), r3.lenbits = P.bits, D) {
                    e3.msg = "invalid literal/lengths set", r3.mode = l;
                    break;
                  }
                  if (r3.distbits = 6, r3.distcode = r3.distdyn, P = { bits: r3.distbits }, D = a(2, r3.lens, r3.nlen, r3.ndist, r3.distcode, 0, r3.work, P), r3.distbits = P.bits, D) {
                    e3.msg = "invalid distances set", r3.mode = l;
                    break;
                  }
                  if (r3.mode = 20, 6 === t3)
                    break e;
                case 20:
                  r3.mode = 21;
                case 21:
                  if (w2 >= 6 && g2 >= 258) {
                    e3.next_out = m2, e3.avail_out = g2, e3.next_in = f2, e3.avail_in = w2, r3.hold = y2, r3.bits = b2, o(e3, E), m2 = e3.next_out, d2 = e3.output, g2 = e3.avail_out, f2 = e3.next_in, p2 = e3.input, w2 = e3.avail_in, y2 = r3.hold, b2 = r3.bits, r3.mode === u && (r3.back = -1);
                    break;
                  }
                  for (r3.back = 0; R = (B = r3.lencode[y2 & (1 << r3.lenbits) - 1]) >>> 16 & 255, N = 65535 & B, !((I = B >>> 24) <= b2); ) {
                    if (0 === w2)
                      break e;
                    w2--, y2 += p2[f2++] << b2, b2 += 8;
                  }
                  if (R && 0 == (240 & R)) {
                    for (C = I, O = R, k = N; R = (B = r3.lencode[k + ((y2 & (1 << C + O) - 1) >> C)]) >>> 16 & 255, N = 65535 & B, !(C + (I = B >>> 24) <= b2); ) {
                      if (0 === w2)
                        break e;
                      w2--, y2 += p2[f2++] << b2, b2 += 8;
                    }
                    y2 >>>= C, b2 -= C, r3.back += C;
                  }
                  if (y2 >>>= I, b2 -= I, r3.back += I, r3.length = N, 0 === R) {
                    r3.mode = 26;
                    break;
                  }
                  if (32 & R) {
                    r3.back = -1, r3.mode = u;
                    break;
                  }
                  if (64 & R) {
                    e3.msg = "invalid literal/length code", r3.mode = l;
                    break;
                  }
                  r3.extra = 15 & R, r3.mode = 22;
                case 22:
                  if (r3.extra) {
                    for (F = r3.extra; b2 < F; ) {
                      if (0 === w2)
                        break e;
                      w2--, y2 += p2[f2++] << b2, b2 += 8;
                    }
                    r3.length += y2 & (1 << r3.extra) - 1, y2 >>>= r3.extra, b2 -= r3.extra, r3.back += r3.extra;
                  }
                  r3.was = r3.length, r3.mode = 23;
                case 23:
                  for (; R = (B = r3.distcode[y2 & (1 << r3.distbits) - 1]) >>> 16 & 255, N = 65535 & B, !((I = B >>> 24) <= b2); ) {
                    if (0 === w2)
                      break e;
                    w2--, y2 += p2[f2++] << b2, b2 += 8;
                  }
                  if (0 == (240 & R)) {
                    for (C = I, O = R, k = N; R = (B = r3.distcode[k + ((y2 & (1 << C + O) - 1) >> C)]) >>> 16 & 255, N = 65535 & B, !(C + (I = B >>> 24) <= b2); ) {
                      if (0 === w2)
                        break e;
                      w2--, y2 += p2[f2++] << b2, b2 += 8;
                    }
                    y2 >>>= C, b2 -= C, r3.back += C;
                  }
                  if (y2 >>>= I, b2 -= I, r3.back += I, 64 & R) {
                    e3.msg = "invalid distance code", r3.mode = l;
                    break;
                  }
                  r3.offset = N, r3.extra = 15 & R, r3.mode = 24;
                case 24:
                  if (r3.extra) {
                    for (F = r3.extra; b2 < F; ) {
                      if (0 === w2)
                        break e;
                      w2--, y2 += p2[f2++] << b2, b2 += 8;
                    }
                    r3.offset += y2 & (1 << r3.extra) - 1, y2 >>>= r3.extra, b2 -= r3.extra, r3.back += r3.extra;
                  }
                  if (r3.offset > r3.dmax) {
                    e3.msg = "invalid distance too far back", r3.mode = l;
                    break;
                  }
                  r3.mode = 25;
                case 25:
                  if (0 === g2)
                    break e;
                  if (T = E - g2, r3.offset > T) {
                    if ((T = r3.offset - T) > r3.whave && r3.sane) {
                      e3.msg = "invalid distance too far back", r3.mode = l;
                      break;
                    }
                    T > r3.wnext ? (T -= r3.wnext, A = r3.wsize - T) : A = r3.wnext - T, T > r3.length && (T = r3.length), S = r3.window;
                  } else
                    S = d2, A = m2 - r3.offset, T = r3.length;
                  T > g2 && (T = g2), g2 -= T, r3.length -= T;
                  do {
                    d2[m2++] = S[A++];
                  } while (--T);
                  0 === r3.length && (r3.mode = 21);
                  break;
                case 26:
                  if (0 === g2)
                    break e;
                  d2[m2++] = r3.length, g2--, r3.mode = 21;
                  break;
                case 27:
                  if (r3.wrap) {
                    for (; b2 < 32; ) {
                      if (0 === w2)
                        break e;
                      w2--, y2 |= p2[f2++] << b2, b2 += 8;
                    }
                    if (E -= g2, e3.total_out += E, r3.total += E, E && (e3.adler = r3.check = r3.flags ? i(r3.check, d2, E, m2 - E) : s(r3.check, d2, E, m2 - E)), E = g2, (r3.flags ? y2 : h(y2)) !== r3.check) {
                      e3.msg = "incorrect data check", r3.mode = l;
                      break;
                    }
                    y2 = 0, b2 = 0;
                  }
                  r3.mode = 28;
                case 28:
                  if (r3.wrap && r3.flags) {
                    for (; b2 < 32; ) {
                      if (0 === w2)
                        break e;
                      w2--, y2 += p2[f2++] << b2, b2 += 8;
                    }
                    if (y2 !== (4294967295 & r3.total)) {
                      e3.msg = "incorrect length check", r3.mode = l;
                      break;
                    }
                    y2 = 0, b2 = 0;
                  }
                  r3.mode = 29;
                case 29:
                  D = 1;
                  break e;
                case l:
                  D = -3;
                  break e;
                case 31:
                  return -4;
                default:
                  return c;
              }
          return e3.next_out = m2, e3.avail_out = g2, e3.next_in = f2, e3.avail_in = w2, r3.hold = y2, r3.bits = b2, (r3.wsize || E !== e3.avail_out && r3.mode < l && (r3.mode < 27 || 4 !== t3)) && v(e3, e3.output, e3.next_out, E - e3.avail_out) ? (r3.mode = 31, -4) : (_ -= e3.avail_in, E -= e3.avail_out, e3.total_in += _, e3.total_out += E, r3.total += E, r3.wrap && E && (e3.adler = r3.check = r3.flags ? i(r3.check, d2, E, e3.next_out - E) : s(r3.check, d2, E, e3.next_out - E)), e3.data_type = r3.bits + (r3.last ? 64 : 0) + (r3.mode === u ? 128 : 0) + (20 === r3.mode || 15 === r3.mode ? 256 : 0), (0 === _ && 0 === E || 4 === t3) && 0 === D && (D = -5), D);
        }, t2.inflateEnd = function(e3) {
          if (!e3 || !e3.state)
            return c;
          var t3 = e3.state;
          return t3.window && (t3.window = null), e3.state = null, 0;
        }, t2.inflateGetHeader = function(e3, t3) {
          var r3;
          return e3 && e3.state ? 0 == (2 & (r3 = e3.state).wrap) ? c : (r3.head = t3, t3.done = false, 0) : c;
        }, t2.inflateSetDictionary = function(e3, t3) {
          var r3, n3 = t3.length;
          return e3 && e3.state ? 0 !== (r3 = e3.state).wrap && 11 !== r3.mode ? c : 11 === r3.mode && s(1, t3, n3, 0) !== r3.check ? -3 : v(e3, t3, n3, n3) ? (r3.mode = 31, -4) : (r3.havedict = 1, 0) : c;
        }, t2.inflateInfo = "pako inflate (from Nodeca project)";
      }, 9241: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(4236), s = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258, 0, 0], i = [16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 16, 72, 78], o = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577, 0, 0], a = [16, 16, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 64, 64];
        e2.exports = function(e3, t3, r3, c, u, l, h, p) {
          var d, f, m, w, g, y, b, x, v, _ = p.bits, E = 0, T = 0, A = 0, S = 0, I = 0, R = 0, N = 0, C = 0, O = 0, k = 0, L = null, D = 0, P = new n2.Buf16(16), F = new n2.Buf16(16), B = null, M = 0;
          for (E = 0; E <= 15; E++)
            P[E] = 0;
          for (T = 0; T < c; T++)
            P[t3[r3 + T]]++;
          for (I = _, S = 15; S >= 1 && 0 === P[S]; S--)
            ;
          if (I > S && (I = S), 0 === S)
            return u[l++] = 20971520, u[l++] = 20971520, p.bits = 1, 0;
          for (A = 1; A < S && 0 === P[A]; A++)
            ;
          for (I < A && (I = A), C = 1, E = 1; E <= 15; E++)
            if (C <<= 1, (C -= P[E]) < 0)
              return -1;
          if (C > 0 && (0 === e3 || 1 !== S))
            return -1;
          for (F[1] = 0, E = 1; E < 15; E++)
            F[E + 1] = F[E] + P[E];
          for (T = 0; T < c; T++)
            0 !== t3[r3 + T] && (h[F[t3[r3 + T]]++] = T);
          if (0 === e3 ? (L = B = h, y = 19) : 1 === e3 ? (L = s, D -= 257, B = i, M -= 257, y = 256) : (L = o, B = a, y = -1), k = 0, T = 0, E = A, g = l, R = I, N = 0, m = -1, w = (O = 1 << I) - 1, 1 === e3 && O > 852 || 2 === e3 && O > 592)
            return 1;
          for (; ; ) {
            b = E - N, h[T] < y ? (x = 0, v = h[T]) : h[T] > y ? (x = B[M + h[T]], v = L[D + h[T]]) : (x = 96, v = 0), d = 1 << E - N, A = f = 1 << R;
            do {
              u[g + (k >> N) + (f -= d)] = b << 24 | x << 16 | v | 0;
            } while (0 !== f);
            for (d = 1 << E - 1; k & d; )
              d >>= 1;
            if (0 !== d ? (k &= d - 1, k += d) : k = 0, T++, 0 == --P[E]) {
              if (E === S)
                break;
              E = t3[r3 + h[T]];
            }
            if (E > I && (k & w) !== m) {
              for (0 === N && (N = I), g += A, C = 1 << (R = E - N); R + N < S && !((C -= P[R + N]) <= 0); )
                R++, C <<= 1;
              if (O += 1 << R, 1 === e3 && O > 852 || 2 === e3 && O > 592)
                return 1;
              u[m = k & w] = I << 24 | R << 16 | g - l | 0;
            }
          }
          return 0 !== k && (u[g + k] = E - N << 24 | 64 << 16 | 0), p.bits = I, 0;
        };
      }, 8898: (e2) => {
        "use strict";
        e2.exports = { 2: "need dictionary", 1: "stream end", 0: "", "-1": "file error", "-2": "stream error", "-3": "data error", "-4": "insufficient memory", "-5": "buffer error", "-6": "incompatible version" };
      }, 342: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(4236);
        function s(e3) {
          for (var t3 = e3.length; --t3 >= 0; )
            e3[t3] = 0;
        }
        var i = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0], o = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13], a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 7], c = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15], u = new Array(576);
        s(u);
        var l = new Array(60);
        s(l);
        var h = new Array(512);
        s(h);
        var p = new Array(256);
        s(p);
        var d = new Array(29);
        s(d);
        var f, m, w, g = new Array(30);
        function y(e3, t3, r3, n3, s2) {
          this.static_tree = e3, this.extra_bits = t3, this.extra_base = r3, this.elems = n3, this.max_length = s2, this.has_stree = e3 && e3.length;
        }
        function b(e3, t3) {
          this.dyn_tree = e3, this.max_code = 0, this.stat_desc = t3;
        }
        function x(e3) {
          return e3 < 256 ? h[e3] : h[256 + (e3 >>> 7)];
        }
        function v(e3, t3) {
          e3.pending_buf[e3.pending++] = 255 & t3, e3.pending_buf[e3.pending++] = t3 >>> 8 & 255;
        }
        function _(e3, t3, r3) {
          e3.bi_valid > 16 - r3 ? (e3.bi_buf |= t3 << e3.bi_valid & 65535, v(e3, e3.bi_buf), e3.bi_buf = t3 >> 16 - e3.bi_valid, e3.bi_valid += r3 - 16) : (e3.bi_buf |= t3 << e3.bi_valid & 65535, e3.bi_valid += r3);
        }
        function E(e3, t3, r3) {
          _(e3, r3[2 * t3], r3[2 * t3 + 1]);
        }
        function T(e3, t3) {
          var r3 = 0;
          do {
            r3 |= 1 & e3, e3 >>>= 1, r3 <<= 1;
          } while (--t3 > 0);
          return r3 >>> 1;
        }
        function A(e3, t3, r3) {
          var n3, s2, i2 = new Array(16), o2 = 0;
          for (n3 = 1; n3 <= 15; n3++)
            i2[n3] = o2 = o2 + r3[n3 - 1] << 1;
          for (s2 = 0; s2 <= t3; s2++) {
            var a2 = e3[2 * s2 + 1];
            0 !== a2 && (e3[2 * s2] = T(i2[a2]++, a2));
          }
        }
        function S(e3) {
          var t3;
          for (t3 = 0; t3 < 286; t3++)
            e3.dyn_ltree[2 * t3] = 0;
          for (t3 = 0; t3 < 30; t3++)
            e3.dyn_dtree[2 * t3] = 0;
          for (t3 = 0; t3 < 19; t3++)
            e3.bl_tree[2 * t3] = 0;
          e3.dyn_ltree[512] = 1, e3.opt_len = e3.static_len = 0, e3.last_lit = e3.matches = 0;
        }
        function I(e3) {
          e3.bi_valid > 8 ? v(e3, e3.bi_buf) : e3.bi_valid > 0 && (e3.pending_buf[e3.pending++] = e3.bi_buf), e3.bi_buf = 0, e3.bi_valid = 0;
        }
        function R(e3, t3, r3, n3) {
          var s2 = 2 * t3, i2 = 2 * r3;
          return e3[s2] < e3[i2] || e3[s2] === e3[i2] && n3[t3] <= n3[r3];
        }
        function N(e3, t3, r3) {
          for (var n3 = e3.heap[r3], s2 = r3 << 1; s2 <= e3.heap_len && (s2 < e3.heap_len && R(t3, e3.heap[s2 + 1], e3.heap[s2], e3.depth) && s2++, !R(t3, n3, e3.heap[s2], e3.depth)); )
            e3.heap[r3] = e3.heap[s2], r3 = s2, s2 <<= 1;
          e3.heap[r3] = n3;
        }
        function C(e3, t3, r3) {
          var n3, s2, a2, c2, u2 = 0;
          if (0 !== e3.last_lit)
            do {
              n3 = e3.pending_buf[e3.d_buf + 2 * u2] << 8 | e3.pending_buf[e3.d_buf + 2 * u2 + 1], s2 = e3.pending_buf[e3.l_buf + u2], u2++, 0 === n3 ? E(e3, s2, t3) : (E(e3, (a2 = p[s2]) + 256 + 1, t3), 0 !== (c2 = i[a2]) && _(e3, s2 -= d[a2], c2), E(e3, a2 = x(--n3), r3), 0 !== (c2 = o[a2]) && _(e3, n3 -= g[a2], c2));
            } while (u2 < e3.last_lit);
          E(e3, 256, t3);
        }
        function O(e3, t3) {
          var r3, n3, s2, i2 = t3.dyn_tree, o2 = t3.stat_desc.static_tree, a2 = t3.stat_desc.has_stree, c2 = t3.stat_desc.elems, u2 = -1;
          for (e3.heap_len = 0, e3.heap_max = 573, r3 = 0; r3 < c2; r3++)
            0 !== i2[2 * r3] ? (e3.heap[++e3.heap_len] = u2 = r3, e3.depth[r3] = 0) : i2[2 * r3 + 1] = 0;
          for (; e3.heap_len < 2; )
            i2[2 * (s2 = e3.heap[++e3.heap_len] = u2 < 2 ? ++u2 : 0)] = 1, e3.depth[s2] = 0, e3.opt_len--, a2 && (e3.static_len -= o2[2 * s2 + 1]);
          for (t3.max_code = u2, r3 = e3.heap_len >> 1; r3 >= 1; r3--)
            N(e3, i2, r3);
          s2 = c2;
          do {
            r3 = e3.heap[1], e3.heap[1] = e3.heap[e3.heap_len--], N(e3, i2, 1), n3 = e3.heap[1], e3.heap[--e3.heap_max] = r3, e3.heap[--e3.heap_max] = n3, i2[2 * s2] = i2[2 * r3] + i2[2 * n3], e3.depth[s2] = (e3.depth[r3] >= e3.depth[n3] ? e3.depth[r3] : e3.depth[n3]) + 1, i2[2 * r3 + 1] = i2[2 * n3 + 1] = s2, e3.heap[1] = s2++, N(e3, i2, 1);
          } while (e3.heap_len >= 2);
          e3.heap[--e3.heap_max] = e3.heap[1], function(e4, t4) {
            var r4, n4, s3, i3, o3, a3, c3 = t4.dyn_tree, u3 = t4.max_code, l2 = t4.stat_desc.static_tree, h2 = t4.stat_desc.has_stree, p2 = t4.stat_desc.extra_bits, d2 = t4.stat_desc.extra_base, f2 = t4.stat_desc.max_length, m2 = 0;
            for (i3 = 0; i3 <= 15; i3++)
              e4.bl_count[i3] = 0;
            for (c3[2 * e4.heap[e4.heap_max] + 1] = 0, r4 = e4.heap_max + 1; r4 < 573; r4++)
              (i3 = c3[2 * c3[2 * (n4 = e4.heap[r4]) + 1] + 1] + 1) > f2 && (i3 = f2, m2++), c3[2 * n4 + 1] = i3, n4 > u3 || (e4.bl_count[i3]++, o3 = 0, n4 >= d2 && (o3 = p2[n4 - d2]), a3 = c3[2 * n4], e4.opt_len += a3 * (i3 + o3), h2 && (e4.static_len += a3 * (l2[2 * n4 + 1] + o3)));
            if (0 !== m2) {
              do {
                for (i3 = f2 - 1; 0 === e4.bl_count[i3]; )
                  i3--;
                e4.bl_count[i3]--, e4.bl_count[i3 + 1] += 2, e4.bl_count[f2]--, m2 -= 2;
              } while (m2 > 0);
              for (i3 = f2; 0 !== i3; i3--)
                for (n4 = e4.bl_count[i3]; 0 !== n4; )
                  (s3 = e4.heap[--r4]) > u3 || (c3[2 * s3 + 1] !== i3 && (e4.opt_len += (i3 - c3[2 * s3 + 1]) * c3[2 * s3], c3[2 * s3 + 1] = i3), n4--);
            }
          }(e3, t3), A(i2, u2, e3.bl_count);
        }
        function k(e3, t3, r3) {
          var n3, s2, i2 = -1, o2 = t3[1], a2 = 0, c2 = 7, u2 = 4;
          for (0 === o2 && (c2 = 138, u2 = 3), t3[2 * (r3 + 1) + 1] = 65535, n3 = 0; n3 <= r3; n3++)
            s2 = o2, o2 = t3[2 * (n3 + 1) + 1], ++a2 < c2 && s2 === o2 || (a2 < u2 ? e3.bl_tree[2 * s2] += a2 : 0 !== s2 ? (s2 !== i2 && e3.bl_tree[2 * s2]++, e3.bl_tree[32]++) : a2 <= 10 ? e3.bl_tree[34]++ : e3.bl_tree[36]++, a2 = 0, i2 = s2, 0 === o2 ? (c2 = 138, u2 = 3) : s2 === o2 ? (c2 = 6, u2 = 3) : (c2 = 7, u2 = 4));
        }
        function L(e3, t3, r3) {
          var n3, s2, i2 = -1, o2 = t3[1], a2 = 0, c2 = 7, u2 = 4;
          for (0 === o2 && (c2 = 138, u2 = 3), n3 = 0; n3 <= r3; n3++)
            if (s2 = o2, o2 = t3[2 * (n3 + 1) + 1], !(++a2 < c2 && s2 === o2)) {
              if (a2 < u2)
                do {
                  E(e3, s2, e3.bl_tree);
                } while (0 != --a2);
              else
                0 !== s2 ? (s2 !== i2 && (E(e3, s2, e3.bl_tree), a2--), E(e3, 16, e3.bl_tree), _(e3, a2 - 3, 2)) : a2 <= 10 ? (E(e3, 17, e3.bl_tree), _(e3, a2 - 3, 3)) : (E(e3, 18, e3.bl_tree), _(e3, a2 - 11, 7));
              a2 = 0, i2 = s2, 0 === o2 ? (c2 = 138, u2 = 3) : s2 === o2 ? (c2 = 6, u2 = 3) : (c2 = 7, u2 = 4);
            }
        }
        s(g);
        var D = false;
        function P(e3, t3, r3, s2) {
          _(e3, 0 + (s2 ? 1 : 0), 3), function(e4, t4, r4, s3) {
            I(e4), v(e4, r4), v(e4, ~r4), n2.arraySet(e4.pending_buf, e4.window, t4, r4, e4.pending), e4.pending += r4;
          }(e3, t3, r3);
        }
        t2._tr_init = function(e3) {
          D || (function() {
            var e4, t3, r3, n3, s2, c2 = new Array(16);
            for (r3 = 0, n3 = 0; n3 < 28; n3++)
              for (d[n3] = r3, e4 = 0; e4 < 1 << i[n3]; e4++)
                p[r3++] = n3;
            for (p[r3 - 1] = n3, s2 = 0, n3 = 0; n3 < 16; n3++)
              for (g[n3] = s2, e4 = 0; e4 < 1 << o[n3]; e4++)
                h[s2++] = n3;
            for (s2 >>= 7; n3 < 30; n3++)
              for (g[n3] = s2 << 7, e4 = 0; e4 < 1 << o[n3] - 7; e4++)
                h[256 + s2++] = n3;
            for (t3 = 0; t3 <= 15; t3++)
              c2[t3] = 0;
            for (e4 = 0; e4 <= 143; )
              u[2 * e4 + 1] = 8, e4++, c2[8]++;
            for (; e4 <= 255; )
              u[2 * e4 + 1] = 9, e4++, c2[9]++;
            for (; e4 <= 279; )
              u[2 * e4 + 1] = 7, e4++, c2[7]++;
            for (; e4 <= 287; )
              u[2 * e4 + 1] = 8, e4++, c2[8]++;
            for (A(u, 287, c2), e4 = 0; e4 < 30; e4++)
              l[2 * e4 + 1] = 5, l[2 * e4] = T(e4, 5);
            f = new y(u, i, 257, 286, 15), m = new y(l, o, 0, 30, 15), w = new y(new Array(0), a, 0, 19, 7);
          }(), D = true), e3.l_desc = new b(e3.dyn_ltree, f), e3.d_desc = new b(e3.dyn_dtree, m), e3.bl_desc = new b(e3.bl_tree, w), e3.bi_buf = 0, e3.bi_valid = 0, S(e3);
        }, t2._tr_stored_block = P, t2._tr_flush_block = function(e3, t3, r3, n3) {
          var s2, i2, o2 = 0;
          e3.level > 0 ? (2 === e3.strm.data_type && (e3.strm.data_type = function(e4) {
            var t4, r4 = 4093624447;
            for (t4 = 0; t4 <= 31; t4++, r4 >>>= 1)
              if (1 & r4 && 0 !== e4.dyn_ltree[2 * t4])
                return 0;
            if (0 !== e4.dyn_ltree[18] || 0 !== e4.dyn_ltree[20] || 0 !== e4.dyn_ltree[26])
              return 1;
            for (t4 = 32; t4 < 256; t4++)
              if (0 !== e4.dyn_ltree[2 * t4])
                return 1;
            return 0;
          }(e3)), O(e3, e3.l_desc), O(e3, e3.d_desc), o2 = function(e4) {
            var t4;
            for (k(e4, e4.dyn_ltree, e4.l_desc.max_code), k(e4, e4.dyn_dtree, e4.d_desc.max_code), O(e4, e4.bl_desc), t4 = 18; t4 >= 3 && 0 === e4.bl_tree[2 * c[t4] + 1]; t4--)
              ;
            return e4.opt_len += 3 * (t4 + 1) + 5 + 5 + 4, t4;
          }(e3), s2 = e3.opt_len + 3 + 7 >>> 3, (i2 = e3.static_len + 3 + 7 >>> 3) <= s2 && (s2 = i2)) : s2 = i2 = r3 + 5, r3 + 4 <= s2 && -1 !== t3 ? P(e3, t3, r3, n3) : 4 === e3.strategy || i2 === s2 ? (_(e3, 2 + (n3 ? 1 : 0), 3), C(e3, u, l)) : (_(e3, 4 + (n3 ? 1 : 0), 3), function(e4, t4, r4, n4) {
            var s3;
            for (_(e4, t4 - 257, 5), _(e4, r4 - 1, 5), _(e4, n4 - 4, 4), s3 = 0; s3 < n4; s3++)
              _(e4, e4.bl_tree[2 * c[s3] + 1], 3);
            L(e4, e4.dyn_ltree, t4 - 1), L(e4, e4.dyn_dtree, r4 - 1);
          }(e3, e3.l_desc.max_code + 1, e3.d_desc.max_code + 1, o2 + 1), C(e3, e3.dyn_ltree, e3.dyn_dtree)), S(e3), n3 && I(e3);
        }, t2._tr_tally = function(e3, t3, r3) {
          return e3.pending_buf[e3.d_buf + 2 * e3.last_lit] = t3 >>> 8 & 255, e3.pending_buf[e3.d_buf + 2 * e3.last_lit + 1] = 255 & t3, e3.pending_buf[e3.l_buf + e3.last_lit] = 255 & r3, e3.last_lit++, 0 === t3 ? e3.dyn_ltree[2 * r3]++ : (e3.matches++, t3--, e3.dyn_ltree[2 * (p[r3] + 256 + 1)]++, e3.dyn_dtree[2 * x(t3)]++), e3.last_lit === e3.lit_bufsize - 1;
        }, t2._tr_align = function(e3) {
          _(e3, 2, 3), E(e3, 256, u), function(e4) {
            16 === e4.bi_valid ? (v(e4, e4.bi_buf), e4.bi_buf = 0, e4.bi_valid = 0) : e4.bi_valid >= 8 && (e4.pending_buf[e4.pending++] = 255 & e4.bi_buf, e4.bi_buf >>= 8, e4.bi_valid -= 8);
          }(e3);
        };
      }, 2292: (e2) => {
        "use strict";
        e2.exports = function() {
          this.input = null, this.next_in = 0, this.avail_in = 0, this.total_in = 0, this.output = null, this.next_out = 0, this.avail_out = 0, this.total_out = 0, this.msg = "", this.state = null, this.data_type = 2, this.adler = 0;
        };
      }, 4155: (e2) => {
        var t2, r2, n2 = e2.exports = {};
        function s() {
          throw new Error("setTimeout has not been defined");
        }
        function i() {
          throw new Error("clearTimeout has not been defined");
        }
        function o(e3) {
          if (t2 === setTimeout)
            return setTimeout(e3, 0);
          if ((t2 === s || !t2) && setTimeout)
            return t2 = setTimeout, setTimeout(e3, 0);
          try {
            return t2(e3, 0);
          } catch (r3) {
            try {
              return t2.call(null, e3, 0);
            } catch (r4) {
              return t2.call(this, e3, 0);
            }
          }
        }
        !function() {
          try {
            t2 = "function" == typeof setTimeout ? setTimeout : s;
          } catch (e3) {
            t2 = s;
          }
          try {
            r2 = "function" == typeof clearTimeout ? clearTimeout : i;
          } catch (e3) {
            r2 = i;
          }
        }();
        var a, c = [], u = false, l = -1;
        function h() {
          u && a && (u = false, a.length ? c = a.concat(c) : l = -1, c.length && p());
        }
        function p() {
          if (!u) {
            var e3 = o(h);
            u = true;
            for (var t3 = c.length; t3; ) {
              for (a = c, c = []; ++l < t3; )
                a && a[l].run();
              l = -1, t3 = c.length;
            }
            a = null, u = false, function(e4) {
              if (r2 === clearTimeout)
                return clearTimeout(e4);
              if ((r2 === i || !r2) && clearTimeout)
                return r2 = clearTimeout, clearTimeout(e4);
              try {
                r2(e4);
              } catch (t4) {
                try {
                  return r2.call(null, e4);
                } catch (t5) {
                  return r2.call(this, e4);
                }
              }
            }(e3);
          }
        }
        function d(e3, t3) {
          this.fun = e3, this.array = t3;
        }
        function f() {
        }
        n2.nextTick = function(e3) {
          var t3 = new Array(arguments.length - 1);
          if (arguments.length > 1)
            for (var r3 = 1; r3 < arguments.length; r3++)
              t3[r3 - 1] = arguments[r3];
          c.push(new d(e3, t3)), 1 !== c.length || u || o(p);
        }, d.prototype.run = function() {
          this.fun.apply(null, this.array);
        }, n2.title = "browser", n2.browser = true, n2.env = {}, n2.argv = [], n2.version = "", n2.versions = {}, n2.on = f, n2.addListener = f, n2.once = f, n2.off = f, n2.removeListener = f, n2.removeAllListeners = f, n2.emit = f, n2.prependListener = f, n2.prependOnceListener = f, n2.listeners = function(e3) {
          return [];
        }, n2.binding = function(e3) {
          throw new Error("process.binding is not supported");
        }, n2.cwd = function() {
          return "/";
        }, n2.chdir = function(e3) {
          throw new Error("process.chdir is not supported");
        }, n2.umask = function() {
          return 0;
        };
      }, 9509: (e2, t2, r2) => {
        var n2 = r2(8764), s = n2.Buffer;
        function i(e3, t3) {
          for (var r3 in e3)
            t3[r3] = e3[r3];
        }
        function o(e3, t3, r3) {
          return s(e3, t3, r3);
        }
        s.from && s.alloc && s.allocUnsafe && s.allocUnsafeSlow ? e2.exports = n2 : (i(n2, t2), t2.Buffer = o), i(s, o), o.from = function(e3, t3, r3) {
          if ("number" == typeof e3)
            throw new TypeError("Argument must not be a number");
          return s(e3, t3, r3);
        }, o.alloc = function(e3, t3, r3) {
          if ("number" != typeof e3)
            throw new TypeError("Argument must be a number");
          var n3 = s(e3);
          return void 0 !== t3 ? "string" == typeof r3 ? n3.fill(t3, r3) : n3.fill(t3) : n3.fill(0), n3;
        }, o.allocUnsafe = function(e3) {
          if ("number" != typeof e3)
            throw new TypeError("Argument must be a number");
          return s(e3);
        }, o.allocUnsafeSlow = function(e3) {
          if ("number" != typeof e3)
            throw new TypeError("Argument must be a number");
          return n2.SlowBuffer(e3);
        };
      }, 6099: (e2, t2, r2) => {
        !function(e3) {
          e3.parser = function(e4, t4) {
            return new s(e4, t4);
          }, e3.SAXParser = s, e3.SAXStream = o, e3.createStream = function(e4, t4) {
            return new o(e4, t4);
          }, e3.MAX_BUFFER_LENGTH = 65536;
          var t3, n2 = ["comment", "sgmlDecl", "textNode", "tagName", "doctype", "procInstName", "procInstBody", "entity", "attribName", "attribValue", "cdata", "script"];
          function s(t4, r3) {
            if (!(this instanceof s))
              return new s(t4, r3);
            var i2 = this;
            !function(e4) {
              for (var t5 = 0, r4 = n2.length; t5 < r4; t5++)
                e4[n2[t5]] = "";
            }(i2), i2.q = i2.c = "", i2.bufferCheckPosition = e3.MAX_BUFFER_LENGTH, i2.opt = r3 || {}, i2.opt.lowercase = i2.opt.lowercase || i2.opt.lowercasetags, i2.looseCase = i2.opt.lowercase ? "toLowerCase" : "toUpperCase", i2.tags = [], i2.closed = i2.closedRoot = i2.sawRoot = false, i2.tag = i2.error = null, i2.strict = !!t4, i2.noscript = !(!t4 && !i2.opt.noscript), i2.state = _.BEGIN, i2.strictEntities = i2.opt.strictEntities, i2.ENTITIES = i2.strictEntities ? Object.create(e3.XML_ENTITIES) : Object.create(e3.ENTITIES), i2.attribList = [], i2.opt.xmlns && (i2.ns = Object.create(u)), i2.trackPosition = false !== i2.opt.position, i2.trackPosition && (i2.position = i2.line = i2.column = 0), T(i2, "onready");
          }
          e3.EVENTS = ["text", "processinginstruction", "sgmldeclaration", "doctype", "comment", "opentagstart", "attribute", "opentag", "closetag", "opencdata", "cdata", "closecdata", "error", "end", "ready", "script", "opennamespace", "closenamespace"], Object.create || (Object.create = function(e4) {
            function t4() {
            }
            return t4.prototype = e4, new t4();
          }), Object.keys || (Object.keys = function(e4) {
            var t4 = [];
            for (var r3 in e4)
              e4.hasOwnProperty(r3) && t4.push(r3);
            return t4;
          }), s.prototype = { end: function() {
            N(this);
          }, write: function(t4) {
            var r3 = this;
            if (this.error)
              throw this.error;
            if (r3.closed)
              return R(r3, "Cannot write after close. Assign an onready handler.");
            if (null === t4)
              return N(r3);
            "object" == typeof t4 && (t4 = t4.toString());
            for (var s2 = 0, i2 = ""; i2 = M(t4, s2++), r3.c = i2, i2; )
              switch (r3.trackPosition && (r3.position++, "\n" === i2 ? (r3.line++, r3.column = 0) : r3.column++), r3.state) {
                case _.BEGIN:
                  if (r3.state = _.BEGIN_WHITESPACE, "\uFEFF" === i2)
                    continue;
                  B(r3, i2);
                  continue;
                case _.BEGIN_WHITESPACE:
                  B(r3, i2);
                  continue;
                case _.TEXT:
                  if (r3.sawRoot && !r3.closedRoot) {
                    for (var o2 = s2 - 1; i2 && "<" !== i2 && "&" !== i2; )
                      (i2 = M(t4, s2++)) && r3.trackPosition && (r3.position++, "\n" === i2 ? (r3.line++, r3.column = 0) : r3.column++);
                    r3.textNode += t4.substring(o2, s2 - 1);
                  }
                  "<" !== i2 || r3.sawRoot && r3.closedRoot && !r3.strict ? (f(i2) || r3.sawRoot && !r3.closedRoot || C(r3, "Text data outside of root node."), "&" === i2 ? r3.state = _.TEXT_ENTITY : r3.textNode += i2) : (r3.state = _.OPEN_WAKA, r3.startTagPosition = r3.position);
                  continue;
                case _.SCRIPT:
                  "<" === i2 ? r3.state = _.SCRIPT_ENDING : r3.script += i2;
                  continue;
                case _.SCRIPT_ENDING:
                  "/" === i2 ? r3.state = _.CLOSE_TAG : (r3.script += "<" + i2, r3.state = _.SCRIPT);
                  continue;
                case _.OPEN_WAKA:
                  if ("!" === i2)
                    r3.state = _.SGML_DECL, r3.sgmlDecl = "";
                  else if (f(i2))
                    ;
                  else if (g(l, i2))
                    r3.state = _.OPEN_TAG, r3.tagName = i2;
                  else if ("/" === i2)
                    r3.state = _.CLOSE_TAG, r3.tagName = "";
                  else if ("?" === i2)
                    r3.state = _.PROC_INST, r3.procInstName = r3.procInstBody = "";
                  else {
                    if (C(r3, "Unencoded <"), r3.startTagPosition + 1 < r3.position) {
                      var a2 = r3.position - r3.startTagPosition;
                      i2 = new Array(a2).join(" ") + i2;
                    }
                    r3.textNode += "<" + i2, r3.state = _.TEXT;
                  }
                  continue;
                case _.SGML_DECL:
                  "[CDATA[" === (r3.sgmlDecl + i2).toUpperCase() ? (A(r3, "onopencdata"), r3.state = _.CDATA, r3.sgmlDecl = "", r3.cdata = "") : r3.sgmlDecl + i2 === "--" ? (r3.state = _.COMMENT, r3.comment = "", r3.sgmlDecl = "") : "DOCTYPE" === (r3.sgmlDecl + i2).toUpperCase() ? (r3.state = _.DOCTYPE, (r3.doctype || r3.sawRoot) && C(r3, "Inappropriately located doctype declaration"), r3.doctype = "", r3.sgmlDecl = "") : ">" === i2 ? (A(r3, "onsgmldeclaration", r3.sgmlDecl), r3.sgmlDecl = "", r3.state = _.TEXT) : m(i2) ? (r3.state = _.SGML_DECL_QUOTED, r3.sgmlDecl += i2) : r3.sgmlDecl += i2;
                  continue;
                case _.SGML_DECL_QUOTED:
                  i2 === r3.q && (r3.state = _.SGML_DECL, r3.q = ""), r3.sgmlDecl += i2;
                  continue;
                case _.DOCTYPE:
                  ">" === i2 ? (r3.state = _.TEXT, A(r3, "ondoctype", r3.doctype), r3.doctype = true) : (r3.doctype += i2, "[" === i2 ? r3.state = _.DOCTYPE_DTD : m(i2) && (r3.state = _.DOCTYPE_QUOTED, r3.q = i2));
                  continue;
                case _.DOCTYPE_QUOTED:
                  r3.doctype += i2, i2 === r3.q && (r3.q = "", r3.state = _.DOCTYPE);
                  continue;
                case _.DOCTYPE_DTD:
                  r3.doctype += i2, "]" === i2 ? r3.state = _.DOCTYPE : m(i2) && (r3.state = _.DOCTYPE_DTD_QUOTED, r3.q = i2);
                  continue;
                case _.DOCTYPE_DTD_QUOTED:
                  r3.doctype += i2, i2 === r3.q && (r3.state = _.DOCTYPE_DTD, r3.q = "");
                  continue;
                case _.COMMENT:
                  "-" === i2 ? r3.state = _.COMMENT_ENDING : r3.comment += i2;
                  continue;
                case _.COMMENT_ENDING:
                  "-" === i2 ? (r3.state = _.COMMENT_ENDED, r3.comment = I(r3.opt, r3.comment), r3.comment && A(r3, "oncomment", r3.comment), r3.comment = "") : (r3.comment += "-" + i2, r3.state = _.COMMENT);
                  continue;
                case _.COMMENT_ENDED:
                  ">" !== i2 ? (C(r3, "Malformed comment"), r3.comment += "--" + i2, r3.state = _.COMMENT) : r3.state = _.TEXT;
                  continue;
                case _.CDATA:
                  "]" === i2 ? r3.state = _.CDATA_ENDING : r3.cdata += i2;
                  continue;
                case _.CDATA_ENDING:
                  "]" === i2 ? r3.state = _.CDATA_ENDING_2 : (r3.cdata += "]" + i2, r3.state = _.CDATA);
                  continue;
                case _.CDATA_ENDING_2:
                  ">" === i2 ? (r3.cdata && A(r3, "oncdata", r3.cdata), A(r3, "onclosecdata"), r3.cdata = "", r3.state = _.TEXT) : "]" === i2 ? r3.cdata += "]" : (r3.cdata += "]]" + i2, r3.state = _.CDATA);
                  continue;
                case _.PROC_INST:
                  "?" === i2 ? r3.state = _.PROC_INST_ENDING : f(i2) ? r3.state = _.PROC_INST_BODY : r3.procInstName += i2;
                  continue;
                case _.PROC_INST_BODY:
                  if (!r3.procInstBody && f(i2))
                    continue;
                  "?" === i2 ? r3.state = _.PROC_INST_ENDING : r3.procInstBody += i2;
                  continue;
                case _.PROC_INST_ENDING:
                  ">" === i2 ? (A(r3, "onprocessinginstruction", { name: r3.procInstName, body: r3.procInstBody }), r3.procInstName = r3.procInstBody = "", r3.state = _.TEXT) : (r3.procInstBody += "?" + i2, r3.state = _.PROC_INST_BODY);
                  continue;
                case _.OPEN_TAG:
                  g(h, i2) ? r3.tagName += i2 : (O(r3), ">" === i2 ? D(r3) : "/" === i2 ? r3.state = _.OPEN_TAG_SLASH : (f(i2) || C(r3, "Invalid character in tag name"), r3.state = _.ATTRIB));
                  continue;
                case _.OPEN_TAG_SLASH:
                  ">" === i2 ? (D(r3, true), P(r3)) : (C(r3, "Forward-slash in opening tag not followed by >"), r3.state = _.ATTRIB);
                  continue;
                case _.ATTRIB:
                  if (f(i2))
                    continue;
                  ">" === i2 ? D(r3) : "/" === i2 ? r3.state = _.OPEN_TAG_SLASH : g(l, i2) ? (r3.attribName = i2, r3.attribValue = "", r3.state = _.ATTRIB_NAME) : C(r3, "Invalid attribute name");
                  continue;
                case _.ATTRIB_NAME:
                  "=" === i2 ? r3.state = _.ATTRIB_VALUE : ">" === i2 ? (C(r3, "Attribute without value"), r3.attribValue = r3.attribName, L(r3), D(r3)) : f(i2) ? r3.state = _.ATTRIB_NAME_SAW_WHITE : g(h, i2) ? r3.attribName += i2 : C(r3, "Invalid attribute name");
                  continue;
                case _.ATTRIB_NAME_SAW_WHITE:
                  if ("=" === i2)
                    r3.state = _.ATTRIB_VALUE;
                  else {
                    if (f(i2))
                      continue;
                    C(r3, "Attribute without value"), r3.tag.attributes[r3.attribName] = "", r3.attribValue = "", A(r3, "onattribute", { name: r3.attribName, value: "" }), r3.attribName = "", ">" === i2 ? D(r3) : g(l, i2) ? (r3.attribName = i2, r3.state = _.ATTRIB_NAME) : (C(r3, "Invalid attribute name"), r3.state = _.ATTRIB);
                  }
                  continue;
                case _.ATTRIB_VALUE:
                  if (f(i2))
                    continue;
                  m(i2) ? (r3.q = i2, r3.state = _.ATTRIB_VALUE_QUOTED) : (C(r3, "Unquoted attribute value"), r3.state = _.ATTRIB_VALUE_UNQUOTED, r3.attribValue = i2);
                  continue;
                case _.ATTRIB_VALUE_QUOTED:
                  if (i2 !== r3.q) {
                    "&" === i2 ? r3.state = _.ATTRIB_VALUE_ENTITY_Q : r3.attribValue += i2;
                    continue;
                  }
                  L(r3), r3.q = "", r3.state = _.ATTRIB_VALUE_CLOSED;
                  continue;
                case _.ATTRIB_VALUE_CLOSED:
                  f(i2) ? r3.state = _.ATTRIB : ">" === i2 ? D(r3) : "/" === i2 ? r3.state = _.OPEN_TAG_SLASH : g(l, i2) ? (C(r3, "No whitespace between attributes"), r3.attribName = i2, r3.attribValue = "", r3.state = _.ATTRIB_NAME) : C(r3, "Invalid attribute name");
                  continue;
                case _.ATTRIB_VALUE_UNQUOTED:
                  if (!w(i2)) {
                    "&" === i2 ? r3.state = _.ATTRIB_VALUE_ENTITY_U : r3.attribValue += i2;
                    continue;
                  }
                  L(r3), ">" === i2 ? D(r3) : r3.state = _.ATTRIB;
                  continue;
                case _.CLOSE_TAG:
                  if (r3.tagName)
                    ">" === i2 ? P(r3) : g(h, i2) ? r3.tagName += i2 : r3.script ? (r3.script += "</" + r3.tagName, r3.tagName = "", r3.state = _.SCRIPT) : (f(i2) || C(r3, "Invalid tagname in closing tag"), r3.state = _.CLOSE_TAG_SAW_WHITE);
                  else {
                    if (f(i2))
                      continue;
                    y(l, i2) ? r3.script ? (r3.script += "</" + i2, r3.state = _.SCRIPT) : C(r3, "Invalid tagname in closing tag.") : r3.tagName = i2;
                  }
                  continue;
                case _.CLOSE_TAG_SAW_WHITE:
                  if (f(i2))
                    continue;
                  ">" === i2 ? P(r3) : C(r3, "Invalid characters in closing tag");
                  continue;
                case _.TEXT_ENTITY:
                case _.ATTRIB_VALUE_ENTITY_Q:
                case _.ATTRIB_VALUE_ENTITY_U:
                  var c2, u2;
                  switch (r3.state) {
                    case _.TEXT_ENTITY:
                      c2 = _.TEXT, u2 = "textNode";
                      break;
                    case _.ATTRIB_VALUE_ENTITY_Q:
                      c2 = _.ATTRIB_VALUE_QUOTED, u2 = "attribValue";
                      break;
                    case _.ATTRIB_VALUE_ENTITY_U:
                      c2 = _.ATTRIB_VALUE_UNQUOTED, u2 = "attribValue";
                  }
                  ";" === i2 ? (r3[u2] += F(r3), r3.entity = "", r3.state = c2) : g(r3.entity.length ? d : p, i2) ? r3.entity += i2 : (C(r3, "Invalid character in entity name"), r3[u2] += "&" + r3.entity + i2, r3.entity = "", r3.state = c2);
                  continue;
                default:
                  throw new Error(r3, "Unknown state: " + r3.state);
              }
            return r3.position >= r3.bufferCheckPosition && function(t5) {
              for (var r4 = Math.max(e3.MAX_BUFFER_LENGTH, 10), s3 = 0, i3 = 0, o3 = n2.length; i3 < o3; i3++) {
                var a3 = t5[n2[i3]].length;
                if (a3 > r4)
                  switch (n2[i3]) {
                    case "textNode":
                      S(t5);
                      break;
                    case "cdata":
                      A(t5, "oncdata", t5.cdata), t5.cdata = "";
                      break;
                    case "script":
                      A(t5, "onscript", t5.script), t5.script = "";
                      break;
                    default:
                      R(t5, "Max buffer length exceeded: " + n2[i3]);
                  }
                s3 = Math.max(s3, a3);
              }
              var c3 = e3.MAX_BUFFER_LENGTH - s3;
              t5.bufferCheckPosition = c3 + t5.position;
            }(r3), r3;
          }, resume: function() {
            return this.error = null, this;
          }, close: function() {
            return this.write(null);
          }, flush: function() {
            var e4;
            S(e4 = this), "" !== e4.cdata && (A(e4, "oncdata", e4.cdata), e4.cdata = ""), "" !== e4.script && (A(e4, "onscript", e4.script), e4.script = "");
          } };
          try {
            t3 = r2(2830).Stream;
          } catch (e4) {
            t3 = function() {
            };
          }
          var i = e3.EVENTS.filter(function(e4) {
            return "error" !== e4 && "end" !== e4;
          });
          function o(e4, r3) {
            if (!(this instanceof o))
              return new o(e4, r3);
            t3.apply(this), this._parser = new s(e4, r3), this.writable = true, this.readable = true;
            var n3 = this;
            this._parser.onend = function() {
              n3.emit("end");
            }, this._parser.onerror = function(e5) {
              n3.emit("error", e5), n3._parser.error = null;
            }, this._decoder = null, i.forEach(function(e5) {
              Object.defineProperty(n3, "on" + e5, { get: function() {
                return n3._parser["on" + e5];
              }, set: function(t4) {
                if (!t4)
                  return n3.removeAllListeners(e5), n3._parser["on" + e5] = t4, t4;
                n3.on(e5, t4);
              }, enumerable: true, configurable: false });
            });
          }
          o.prototype = Object.create(t3.prototype, { constructor: { value: o } }), o.prototype.write = function(e4) {
            if ("function" == typeof Buffer && "function" == typeof Buffer.isBuffer && Buffer.isBuffer(e4)) {
              if (!this._decoder) {
                var t4 = r2(2553).s;
                this._decoder = new t4("utf8");
              }
              e4 = this._decoder.write(e4);
            }
            return this._parser.write(e4.toString()), this.emit("data", e4), true;
          }, o.prototype.end = function(e4) {
            return e4 && e4.length && this.write(e4), this._parser.end(), true;
          }, o.prototype.on = function(e4, r3) {
            var n3 = this;
            return n3._parser["on" + e4] || -1 === i.indexOf(e4) || (n3._parser["on" + e4] = function() {
              var t4 = 1 === arguments.length ? [arguments[0]] : Array.apply(null, arguments);
              t4.splice(0, 0, e4), n3.emit.apply(n3, t4);
            }), t3.prototype.on.call(n3, e4, r3);
          };
          var a = "http://www.w3.org/XML/1998/namespace", c = "http://www.w3.org/2000/xmlns/", u = { xml: a, xmlns: c }, l = /[:_A-Za-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD]/, h = /[:_A-Za-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD\u00B7\u0300-\u036F\u203F-\u2040.\d-]/, p = /[#:_A-Za-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD]/, d = /[#:_A-Za-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD\u00B7\u0300-\u036F\u203F-\u2040.\d-]/;
          function f(e4) {
            return " " === e4 || "\n" === e4 || "\r" === e4 || "	" === e4;
          }
          function m(e4) {
            return '"' === e4 || "'" === e4;
          }
          function w(e4) {
            return ">" === e4 || f(e4);
          }
          function g(e4, t4) {
            return e4.test(t4);
          }
          function y(e4, t4) {
            return !g(e4, t4);
          }
          var b, x, v, _ = 0;
          for (var E in e3.STATE = { BEGIN: _++, BEGIN_WHITESPACE: _++, TEXT: _++, TEXT_ENTITY: _++, OPEN_WAKA: _++, SGML_DECL: _++, SGML_DECL_QUOTED: _++, DOCTYPE: _++, DOCTYPE_QUOTED: _++, DOCTYPE_DTD: _++, DOCTYPE_DTD_QUOTED: _++, COMMENT_STARTING: _++, COMMENT: _++, COMMENT_ENDING: _++, COMMENT_ENDED: _++, CDATA: _++, CDATA_ENDING: _++, CDATA_ENDING_2: _++, PROC_INST: _++, PROC_INST_BODY: _++, PROC_INST_ENDING: _++, OPEN_TAG: _++, OPEN_TAG_SLASH: _++, ATTRIB: _++, ATTRIB_NAME: _++, ATTRIB_NAME_SAW_WHITE: _++, ATTRIB_VALUE: _++, ATTRIB_VALUE_QUOTED: _++, ATTRIB_VALUE_CLOSED: _++, ATTRIB_VALUE_UNQUOTED: _++, ATTRIB_VALUE_ENTITY_Q: _++, ATTRIB_VALUE_ENTITY_U: _++, CLOSE_TAG: _++, CLOSE_TAG_SAW_WHITE: _++, SCRIPT: _++, SCRIPT_ENDING: _++ }, e3.XML_ENTITIES = { amp: "&", gt: ">", lt: "<", quot: '"', apos: "'" }, e3.ENTITIES = { amp: "&", gt: ">", lt: "<", quot: '"', apos: "'", AElig: 198, Aacute: 193, Acirc: 194, Agrave: 192, Aring: 197, Atilde: 195, Auml: 196, Ccedil: 199, ETH: 208, Eacute: 201, Ecirc: 202, Egrave: 200, Euml: 203, Iacute: 205, Icirc: 206, Igrave: 204, Iuml: 207, Ntilde: 209, Oacute: 211, Ocirc: 212, Ograve: 210, Oslash: 216, Otilde: 213, Ouml: 214, THORN: 222, Uacute: 218, Ucirc: 219, Ugrave: 217, Uuml: 220, Yacute: 221, aacute: 225, acirc: 226, aelig: 230, agrave: 224, aring: 229, atilde: 227, auml: 228, ccedil: 231, eacute: 233, ecirc: 234, egrave: 232, eth: 240, euml: 235, iacute: 237, icirc: 238, igrave: 236, iuml: 239, ntilde: 241, oacute: 243, ocirc: 244, ograve: 242, oslash: 248, otilde: 245, ouml: 246, szlig: 223, thorn: 254, uacute: 250, ucirc: 251, ugrave: 249, uuml: 252, yacute: 253, yuml: 255, copy: 169, reg: 174, nbsp: 160, iexcl: 161, cent: 162, pound: 163, curren: 164, yen: 165, brvbar: 166, sect: 167, uml: 168, ordf: 170, laquo: 171, not: 172, shy: 173, macr: 175, deg: 176, plusmn: 177, sup1: 185, sup2: 178, sup3: 179, acute: 180, micro: 181, para: 182, middot: 183, cedil: 184, ordm: 186, raquo: 187, frac14: 188, frac12: 189, frac34: 190, iquest: 191, times: 215, divide: 247, OElig: 338, oelig: 339, Scaron: 352, scaron: 353, Yuml: 376, fnof: 402, circ: 710, tilde: 732, Alpha: 913, Beta: 914, Gamma: 915, Delta: 916, Epsilon: 917, Zeta: 918, Eta: 919, Theta: 920, Iota: 921, Kappa: 922, Lambda: 923, Mu: 924, Nu: 925, Xi: 926, Omicron: 927, Pi: 928, Rho: 929, Sigma: 931, Tau: 932, Upsilon: 933, Phi: 934, Chi: 935, Psi: 936, Omega: 937, alpha: 945, beta: 946, gamma: 947, delta: 948, epsilon: 949, zeta: 950, eta: 951, theta: 952, iota: 953, kappa: 954, lambda: 955, mu: 956, nu: 957, xi: 958, omicron: 959, pi: 960, rho: 961, sigmaf: 962, sigma: 963, tau: 964, upsilon: 965, phi: 966, chi: 967, psi: 968, omega: 969, thetasym: 977, upsih: 978, piv: 982, ensp: 8194, emsp: 8195, thinsp: 8201, zwnj: 8204, zwj: 8205, lrm: 8206, rlm: 8207, ndash: 8211, mdash: 8212, lsquo: 8216, rsquo: 8217, sbquo: 8218, ldquo: 8220, rdquo: 8221, bdquo: 8222, dagger: 8224, Dagger: 8225, bull: 8226, hellip: 8230, permil: 8240, prime: 8242, Prime: 8243, lsaquo: 8249, rsaquo: 8250, oline: 8254, frasl: 8260, euro: 8364, image: 8465, weierp: 8472, real: 8476, trade: 8482, alefsym: 8501, larr: 8592, uarr: 8593, rarr: 8594, darr: 8595, harr: 8596, crarr: 8629, lArr: 8656, uArr: 8657, rArr: 8658, dArr: 8659, hArr: 8660, forall: 8704, part: 8706, exist: 8707, empty: 8709, nabla: 8711, isin: 8712, notin: 8713, ni: 8715, prod: 8719, sum: 8721, minus: 8722, lowast: 8727, radic: 8730, prop: 8733, infin: 8734, ang: 8736, and: 8743, or: 8744, cap: 8745, cup: 8746, int: 8747, there4: 8756, sim: 8764, cong: 8773, asymp: 8776, ne: 8800, equiv: 8801, le: 8804, ge: 8805, sub: 8834, sup: 8835, nsub: 8836, sube: 8838, supe: 8839, oplus: 8853, otimes: 8855, perp: 8869, sdot: 8901, lceil: 8968, rceil: 8969, lfloor: 8970, rfloor: 8971, lang: 9001, rang: 9002, loz: 9674, spades: 9824, clubs: 9827, hearts: 9829, diams: 9830 }, Object.keys(e3.ENTITIES).forEach(function(t4) {
            var r3 = e3.ENTITIES[t4], n3 = "number" == typeof r3 ? String.fromCharCode(r3) : r3;
            e3.ENTITIES[t4] = n3;
          }), e3.STATE)
            e3.STATE[e3.STATE[E]] = E;
          function T(e4, t4, r3) {
            e4[t4] && e4[t4](r3);
          }
          function A(e4, t4, r3) {
            e4.textNode && S(e4), T(e4, t4, r3);
          }
          function S(e4) {
            e4.textNode = I(e4.opt, e4.textNode), e4.textNode && T(e4, "ontext", e4.textNode), e4.textNode = "";
          }
          function I(e4, t4) {
            return e4.trim && (t4 = t4.trim()), e4.normalize && (t4 = t4.replace(/\s+/g, " ")), t4;
          }
          function R(e4, t4) {
            return S(e4), e4.trackPosition && (t4 += "\nLine: " + e4.line + "\nColumn: " + e4.column + "\nChar: " + e4.c), t4 = new Error(t4), e4.error = t4, T(e4, "onerror", t4), e4;
          }
          function N(e4) {
            return e4.sawRoot && !e4.closedRoot && C(e4, "Unclosed root tag"), e4.state !== _.BEGIN && e4.state !== _.BEGIN_WHITESPACE && e4.state !== _.TEXT && R(e4, "Unexpected end"), S(e4), e4.c = "", e4.closed = true, T(e4, "onend"), s.call(e4, e4.strict, e4.opt), e4;
          }
          function C(e4, t4) {
            if ("object" != typeof e4 || !(e4 instanceof s))
              throw new Error("bad call to strictFail");
            e4.strict && R(e4, t4);
          }
          function O(e4) {
            e4.strict || (e4.tagName = e4.tagName[e4.looseCase]());
            var t4 = e4.tags[e4.tags.length - 1] || e4, r3 = e4.tag = { name: e4.tagName, attributes: {} };
            e4.opt.xmlns && (r3.ns = t4.ns), e4.attribList.length = 0, A(e4, "onopentagstart", r3);
          }
          function k(e4, t4) {
            var r3 = e4.indexOf(":") < 0 ? ["", e4] : e4.split(":"), n3 = r3[0], s2 = r3[1];
            return t4 && "xmlns" === e4 && (n3 = "xmlns", s2 = ""), { prefix: n3, local: s2 };
          }
          function L(e4) {
            if (e4.strict || (e4.attribName = e4.attribName[e4.looseCase]()), -1 !== e4.attribList.indexOf(e4.attribName) || e4.tag.attributes.hasOwnProperty(e4.attribName))
              e4.attribName = e4.attribValue = "";
            else {
              if (e4.opt.xmlns) {
                var t4 = k(e4.attribName, true), r3 = t4.prefix, n3 = t4.local;
                if ("xmlns" === r3)
                  if ("xml" === n3 && e4.attribValue !== a)
                    C(e4, "xml: prefix must be bound to " + a + "\nActual: " + e4.attribValue);
                  else if ("xmlns" === n3 && e4.attribValue !== c)
                    C(e4, "xmlns: prefix must be bound to " + c + "\nActual: " + e4.attribValue);
                  else {
                    var s2 = e4.tag, i2 = e4.tags[e4.tags.length - 1] || e4;
                    s2.ns === i2.ns && (s2.ns = Object.create(i2.ns)), s2.ns[n3] = e4.attribValue;
                  }
                e4.attribList.push([e4.attribName, e4.attribValue]);
              } else
                e4.tag.attributes[e4.attribName] = e4.attribValue, A(e4, "onattribute", { name: e4.attribName, value: e4.attribValue });
              e4.attribName = e4.attribValue = "";
            }
          }
          function D(e4, t4) {
            if (e4.opt.xmlns) {
              var r3 = e4.tag, n3 = k(e4.tagName);
              r3.prefix = n3.prefix, r3.local = n3.local, r3.uri = r3.ns[n3.prefix] || "", r3.prefix && !r3.uri && (C(e4, "Unbound namespace prefix: " + JSON.stringify(e4.tagName)), r3.uri = n3.prefix);
              var s2 = e4.tags[e4.tags.length - 1] || e4;
              r3.ns && s2.ns !== r3.ns && Object.keys(r3.ns).forEach(function(t5) {
                A(e4, "onopennamespace", { prefix: t5, uri: r3.ns[t5] });
              });
              for (var i2 = 0, o2 = e4.attribList.length; i2 < o2; i2++) {
                var a2 = e4.attribList[i2], c2 = a2[0], u2 = a2[1], l2 = k(c2, true), h2 = l2.prefix, p2 = l2.local, d2 = "" === h2 ? "" : r3.ns[h2] || "", f2 = { name: c2, value: u2, prefix: h2, local: p2, uri: d2 };
                h2 && "xmlns" !== h2 && !d2 && (C(e4, "Unbound namespace prefix: " + JSON.stringify(h2)), f2.uri = h2), e4.tag.attributes[c2] = f2, A(e4, "onattribute", f2);
              }
              e4.attribList.length = 0;
            }
            e4.tag.isSelfClosing = !!t4, e4.sawRoot = true, e4.tags.push(e4.tag), A(e4, "onopentag", e4.tag), t4 || (e4.noscript || "script" !== e4.tagName.toLowerCase() ? e4.state = _.TEXT : e4.state = _.SCRIPT, e4.tag = null, e4.tagName = ""), e4.attribName = e4.attribValue = "", e4.attribList.length = 0;
          }
          function P(e4) {
            if (!e4.tagName)
              return C(e4, "Weird empty close tag."), e4.textNode += "</>", void (e4.state = _.TEXT);
            if (e4.script) {
              if ("script" !== e4.tagName)
                return e4.script += "</" + e4.tagName + ">", e4.tagName = "", void (e4.state = _.SCRIPT);
              A(e4, "onscript", e4.script), e4.script = "";
            }
            var t4 = e4.tags.length, r3 = e4.tagName;
            e4.strict || (r3 = r3[e4.looseCase]());
            for (var n3 = r3; t4-- && e4.tags[t4].name !== n3; )
              C(e4, "Unexpected close tag");
            if (t4 < 0)
              return C(e4, "Unmatched closing tag: " + e4.tagName), e4.textNode += "</" + e4.tagName + ">", void (e4.state = _.TEXT);
            e4.tagName = r3;
            for (var s2 = e4.tags.length; s2-- > t4; ) {
              var i2 = e4.tag = e4.tags.pop();
              e4.tagName = e4.tag.name, A(e4, "onclosetag", e4.tagName);
              var o2 = {};
              for (var a2 in i2.ns)
                o2[a2] = i2.ns[a2];
              var c2 = e4.tags[e4.tags.length - 1] || e4;
              e4.opt.xmlns && i2.ns !== c2.ns && Object.keys(i2.ns).forEach(function(t5) {
                var r4 = i2.ns[t5];
                A(e4, "onclosenamespace", { prefix: t5, uri: r4 });
              });
            }
            0 === t4 && (e4.closedRoot = true), e4.tagName = e4.attribValue = e4.attribName = "", e4.attribList.length = 0, e4.state = _.TEXT;
          }
          function F(e4) {
            var t4, r3 = e4.entity, n3 = r3.toLowerCase(), s2 = "";
            return e4.ENTITIES[r3] ? e4.ENTITIES[r3] : e4.ENTITIES[n3] ? e4.ENTITIES[n3] : ("#" === (r3 = n3).charAt(0) && ("x" === r3.charAt(1) ? (r3 = r3.slice(2), s2 = (t4 = parseInt(r3, 16)).toString(16)) : (r3 = r3.slice(1), s2 = (t4 = parseInt(r3, 10)).toString(10))), r3 = r3.replace(/^0+/, ""), isNaN(t4) || s2.toLowerCase() !== r3 ? (C(e4, "Invalid character entity"), "&" + e4.entity + ";") : String.fromCodePoint(t4));
          }
          function B(e4, t4) {
            "<" === t4 ? (e4.state = _.OPEN_WAKA, e4.startTagPosition = e4.position) : f(t4) || (C(e4, "Non-whitespace before first tag."), e4.textNode = t4, e4.state = _.TEXT);
          }
          function M(e4, t4) {
            var r3 = "";
            return t4 < e4.length && (r3 = e4.charAt(t4)), r3;
          }
          _ = e3.STATE, String.fromCodePoint || (b = String.fromCharCode, x = Math.floor, v = function() {
            var e4, t4, r3 = 16384, n3 = [], s2 = -1, i2 = arguments.length;
            if (!i2)
              return "";
            for (var o2 = ""; ++s2 < i2; ) {
              var a2 = Number(arguments[s2]);
              if (!isFinite(a2) || a2 < 0 || a2 > 1114111 || x(a2) !== a2)
                throw RangeError("Invalid code point: " + a2);
              a2 <= 65535 ? n3.push(a2) : (e4 = 55296 + ((a2 -= 65536) >> 10), t4 = a2 % 1024 + 56320, n3.push(e4, t4)), (s2 + 1 === i2 || n3.length > r3) && (o2 += b.apply(null, n3), n3.length = 0);
            }
            return o2;
          }, Object.defineProperty ? Object.defineProperty(String, "fromCodePoint", { value: v, configurable: true, writable: true }) : String.fromCodePoint = v);
        }(t2);
      }, 4889: function(e2, t2, r2) {
        var n2 = r2(4155);
        !function(e3, t3) {
          "use strict";
          if (!e3.setImmediate) {
            var r3, s, i, o, a, c = 1, u = {}, l = false, h = e3.document, p = Object.getPrototypeOf && Object.getPrototypeOf(e3);
            p = p && p.setTimeout ? p : e3, "[object process]" === {}.toString.call(e3.process) ? r3 = function(e4) {
              n2.nextTick(function() {
                f(e4);
              });
            } : function() {
              if (e3.postMessage && !e3.importScripts) {
                var t4 = true, r4 = e3.onmessage;
                return e3.onmessage = function() {
                  t4 = false;
                }, e3.postMessage("", "*"), e3.onmessage = r4, t4;
              }
            }() ? (o = "setImmediate$" + Math.random() + "$", a = function(t4) {
              t4.source === e3 && "string" == typeof t4.data && 0 === t4.data.indexOf(o) && f(+t4.data.slice(o.length));
            }, e3.addEventListener ? e3.addEventListener("message", a, false) : e3.attachEvent("onmessage", a), r3 = function(t4) {
              e3.postMessage(o + t4, "*");
            }) : e3.MessageChannel ? ((i = new MessageChannel()).port1.onmessage = function(e4) {
              f(e4.data);
            }, r3 = function(e4) {
              i.port2.postMessage(e4);
            }) : h && "onreadystatechange" in h.createElement("script") ? (s = h.documentElement, r3 = function(e4) {
              var t4 = h.createElement("script");
              t4.onreadystatechange = function() {
                f(e4), t4.onreadystatechange = null, s.removeChild(t4), t4 = null;
              }, s.appendChild(t4);
            }) : r3 = function(e4) {
              setTimeout(f, 0, e4);
            }, p.setImmediate = function(e4) {
              "function" != typeof e4 && (e4 = new Function("" + e4));
              for (var t4 = new Array(arguments.length - 1), n3 = 0; n3 < t4.length; n3++)
                t4[n3] = arguments[n3 + 1];
              var s2 = { callback: e4, args: t4 };
              return u[c] = s2, r3(c), c++;
            }, p.clearImmediate = d;
          }
          function d(e4) {
            delete u[e4];
          }
          function f(e4) {
            if (l)
              setTimeout(f, 0, e4);
            else {
              var t4 = u[e4];
              if (t4) {
                l = true;
                try {
                  !function(e5) {
                    var t5 = e5.callback, r4 = e5.args;
                    switch (r4.length) {
                      case 0:
                        t5();
                        break;
                      case 1:
                        t5(r4[0]);
                        break;
                      case 2:
                        t5(r4[0], r4[1]);
                        break;
                      case 3:
                        t5(r4[0], r4[1], r4[2]);
                        break;
                      default:
                        t5.apply(void 0, r4);
                    }
                  }(t4);
                } finally {
                  d(e4), l = false;
                }
              }
            }
          }
        }("undefined" == typeof self ? void 0 === r2.g ? this : r2.g : self);
      }, 2830: (e2, t2, r2) => {
        e2.exports = s;
        var n2 = r2(7187).EventEmitter;
        function s() {
          n2.call(this);
        }
        r2(5717)(s, n2), s.Readable = r2(6577), s.Writable = r2(323), s.Duplex = r2(8656), s.Transform = r2(4473), s.PassThrough = r2(2366), s.finished = r2(1086), s.pipeline = r2(6472), s.Stream = s, s.prototype.pipe = function(e3, t3) {
          var r3 = this;
          function s2(t4) {
            e3.writable && false === e3.write(t4) && r3.pause && r3.pause();
          }
          function i() {
            r3.readable && r3.resume && r3.resume();
          }
          r3.on("data", s2), e3.on("drain", i), e3._isStdio || t3 && false === t3.end || (r3.on("end", a), r3.on("close", c));
          var o = false;
          function a() {
            o || (o = true, e3.end());
          }
          function c() {
            o || (o = true, "function" == typeof e3.destroy && e3.destroy());
          }
          function u(e4) {
            if (l(), 0 === n2.listenerCount(this, "error"))
              throw e4;
          }
          function l() {
            r3.removeListener("data", s2), e3.removeListener("drain", i), r3.removeListener("end", a), r3.removeListener("close", c), r3.removeListener("error", u), e3.removeListener("error", u), r3.removeListener("end", l), r3.removeListener("close", l), e3.removeListener("close", l);
          }
          return r3.on("error", u), e3.on("error", u), r3.on("end", l), r3.on("close", l), e3.on("close", l), e3.emit("pipe", r3), e3;
        };
      }, 8106: (e2) => {
        "use strict";
        var t2 = {};
        function r2(e3, r3, n3) {
          n3 || (n3 = Error);
          var s = function(e4) {
            var t3, n4;
            function s2(t4, n5, s3) {
              return e4.call(this, function(e5, t5, n6) {
                return "string" == typeof r3 ? r3 : r3(e5, t5, n6);
              }(t4, n5, s3)) || this;
            }
            return n4 = e4, (t3 = s2).prototype = Object.create(n4.prototype), t3.prototype.constructor = t3, t3.__proto__ = n4, s2;
          }(n3);
          s.prototype.name = n3.name, s.prototype.code = e3, t2[e3] = s;
        }
        function n2(e3, t3) {
          if (Array.isArray(e3)) {
            var r3 = e3.length;
            return e3 = e3.map(function(e4) {
              return String(e4);
            }), r3 > 2 ? "one of ".concat(t3, " ").concat(e3.slice(0, r3 - 1).join(", "), ", or ") + e3[r3 - 1] : 2 === r3 ? "one of ".concat(t3, " ").concat(e3[0], " or ").concat(e3[1]) : "of ".concat(t3, " ").concat(e3[0]);
          }
          return "of ".concat(t3, " ").concat(String(e3));
        }
        r2("ERR_INVALID_OPT_VALUE", function(e3, t3) {
          return 'The value "' + t3 + '" is invalid for option "' + e3 + '"';
        }, TypeError), r2("ERR_INVALID_ARG_TYPE", function(e3, t3, r3) {
          var s, i, o, a, c;
          if ("string" == typeof t3 && (i = "not ", t3.substr(0, i.length) === i) ? (s = "must not be", t3 = t3.replace(/^not /, "")) : s = "must be", function(e4, t4, r4) {
            return (void 0 === r4 || r4 > e4.length) && (r4 = e4.length), e4.substring(r4 - t4.length, r4) === t4;
          }(e3, " argument"))
            o = "The ".concat(e3, " ").concat(s, " ").concat(n2(t3, "type"));
          else {
            var u = ("number" != typeof c && (c = 0), c + ".".length > (a = e3).length || -1 === a.indexOf(".", c) ? "argument" : "property");
            o = 'The "'.concat(e3, '" ').concat(u, " ").concat(s, " ").concat(n2(t3, "type"));
          }
          return o + ". Received type ".concat(typeof r3);
        }, TypeError), r2("ERR_STREAM_PUSH_AFTER_EOF", "stream.push() after EOF"), r2("ERR_METHOD_NOT_IMPLEMENTED", function(e3) {
          return "The " + e3 + " method is not implemented";
        }), r2("ERR_STREAM_PREMATURE_CLOSE", "Premature close"), r2("ERR_STREAM_DESTROYED", function(e3) {
          return "Cannot call " + e3 + " after a stream was destroyed";
        }), r2("ERR_MULTIPLE_CALLBACK", "Callback called multiple times"), r2("ERR_STREAM_CANNOT_PIPE", "Cannot pipe, not readable"), r2("ERR_STREAM_WRITE_AFTER_END", "write after end"), r2("ERR_STREAM_NULL_VALUES", "May not write null values to stream", TypeError), r2("ERR_UNKNOWN_ENCODING", function(e3) {
          return "Unknown encoding: " + e3;
        }, TypeError), r2("ERR_STREAM_UNSHIFT_AFTER_END_EVENT", "stream.unshift() after end event"), e2.exports.q = t2;
      }, 8656: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(4155), s = Object.keys || function(e3) {
          var t3 = [];
          for (var r3 in e3)
            t3.push(r3);
          return t3;
        };
        e2.exports = l;
        var i = r2(6577), o = r2(323);
        r2(5717)(l, i);
        for (var a = s(o.prototype), c = 0; c < a.length; c++) {
          var u = a[c];
          l.prototype[u] || (l.prototype[u] = o.prototype[u]);
        }
        function l(e3) {
          if (!(this instanceof l))
            return new l(e3);
          i.call(this, e3), o.call(this, e3), this.allowHalfOpen = true, e3 && (false === e3.readable && (this.readable = false), false === e3.writable && (this.writable = false), false === e3.allowHalfOpen && (this.allowHalfOpen = false, this.once("end", h)));
        }
        function h() {
          this._writableState.ended || n2.nextTick(p, this);
        }
        function p(e3) {
          e3.end();
        }
        Object.defineProperty(l.prototype, "writableHighWaterMark", { enumerable: false, get: function() {
          return this._writableState.highWaterMark;
        } }), Object.defineProperty(l.prototype, "writableBuffer", { enumerable: false, get: function() {
          return this._writableState && this._writableState.getBuffer();
        } }), Object.defineProperty(l.prototype, "writableLength", { enumerable: false, get: function() {
          return this._writableState.length;
        } }), Object.defineProperty(l.prototype, "destroyed", { enumerable: false, get: function() {
          return void 0 !== this._readableState && void 0 !== this._writableState && this._readableState.destroyed && this._writableState.destroyed;
        }, set: function(e3) {
          void 0 !== this._readableState && void 0 !== this._writableState && (this._readableState.destroyed = e3, this._writableState.destroyed = e3);
        } });
      }, 2366: (e2, t2, r2) => {
        "use strict";
        e2.exports = s;
        var n2 = r2(4473);
        function s(e3) {
          if (!(this instanceof s))
            return new s(e3);
          n2.call(this, e3);
        }
        r2(5717)(s, n2), s.prototype._transform = function(e3, t3, r3) {
          r3(null, e3);
        };
      }, 6577: (e2, t2, r2) => {
        "use strict";
        var n2, s = r2(4155);
        e2.exports = A, A.ReadableState = T, r2(7187).EventEmitter;
        var i, o = function(e3, t3) {
          return e3.listeners(t3).length;
        }, a = r2(3194), c = r2(8764).Buffer, u = r2.g.Uint8Array || function() {
        }, l = r2(5575);
        i = l && l.debuglog ? l.debuglog("stream") : function() {
        };
        var h, p, d, f = r2(9686), m = r2(1029), w = r2(94).getHighWaterMark, g = r2(8106).q, y = g.ERR_INVALID_ARG_TYPE, b = g.ERR_STREAM_PUSH_AFTER_EOF, x = g.ERR_METHOD_NOT_IMPLEMENTED, v = g.ERR_STREAM_UNSHIFT_AFTER_END_EVENT;
        r2(5717)(A, a);
        var _ = m.errorOrDestroy, E = ["error", "close", "destroy", "pause", "resume"];
        function T(e3, t3, s2) {
          n2 = n2 || r2(8656), e3 = e3 || {}, "boolean" != typeof s2 && (s2 = t3 instanceof n2), this.objectMode = !!e3.objectMode, s2 && (this.objectMode = this.objectMode || !!e3.readableObjectMode), this.highWaterMark = w(this, e3, "readableHighWaterMark", s2), this.buffer = new f(), this.length = 0, this.pipes = null, this.pipesCount = 0, this.flowing = null, this.ended = false, this.endEmitted = false, this.reading = false, this.sync = true, this.needReadable = false, this.emittedReadable = false, this.readableListening = false, this.resumeScheduled = false, this.paused = true, this.emitClose = false !== e3.emitClose, this.autoDestroy = !!e3.autoDestroy, this.destroyed = false, this.defaultEncoding = e3.defaultEncoding || "utf8", this.awaitDrain = 0, this.readingMore = false, this.decoder = null, this.encoding = null, e3.encoding && (h || (h = r2(2553).s), this.decoder = new h(e3.encoding), this.encoding = e3.encoding);
        }
        function A(e3) {
          if (n2 = n2 || r2(8656), !(this instanceof A))
            return new A(e3);
          var t3 = this instanceof n2;
          this._readableState = new T(e3, this, t3), this.readable = true, e3 && ("function" == typeof e3.read && (this._read = e3.read), "function" == typeof e3.destroy && (this._destroy = e3.destroy)), a.call(this);
        }
        function S(e3, t3, r3, n3, s2) {
          i("readableAddChunk", t3);
          var o2, a2 = e3._readableState;
          if (null === t3)
            a2.reading = false, function(e4, t4) {
              if (i("onEofChunk"), !t4.ended) {
                if (t4.decoder) {
                  var r4 = t4.decoder.end();
                  r4 && r4.length && (t4.buffer.push(r4), t4.length += t4.objectMode ? 1 : r4.length);
                }
                t4.ended = true, t4.sync ? C(e4) : (t4.needReadable = false, t4.emittedReadable || (t4.emittedReadable = true, O(e4)));
              }
            }(e3, a2);
          else if (s2 || (o2 = function(e4, t4) {
            var r4, n4;
            return n4 = t4, c.isBuffer(n4) || n4 instanceof u || "string" == typeof t4 || void 0 === t4 || e4.objectMode || (r4 = new y("chunk", ["string", "Buffer", "Uint8Array"], t4)), r4;
          }(a2, t3)), o2)
            _(e3, o2);
          else if (a2.objectMode || t3 && t3.length > 0)
            if ("string" == typeof t3 || a2.objectMode || Object.getPrototypeOf(t3) === c.prototype || (t3 = function(e4) {
              return c.from(e4);
            }(t3)), n3)
              a2.endEmitted ? _(e3, new v()) : I(e3, a2, t3, true);
            else if (a2.ended)
              _(e3, new b());
            else {
              if (a2.destroyed)
                return false;
              a2.reading = false, a2.decoder && !r3 ? (t3 = a2.decoder.write(t3), a2.objectMode || 0 !== t3.length ? I(e3, a2, t3, false) : k(e3, a2)) : I(e3, a2, t3, false);
            }
          else
            n3 || (a2.reading = false, k(e3, a2));
          return !a2.ended && (a2.length < a2.highWaterMark || 0 === a2.length);
        }
        function I(e3, t3, r3, n3) {
          t3.flowing && 0 === t3.length && !t3.sync ? (t3.awaitDrain = 0, e3.emit("data", r3)) : (t3.length += t3.objectMode ? 1 : r3.length, n3 ? t3.buffer.unshift(r3) : t3.buffer.push(r3), t3.needReadable && C(e3)), k(e3, t3);
        }
        Object.defineProperty(A.prototype, "destroyed", { enumerable: false, get: function() {
          return void 0 !== this._readableState && this._readableState.destroyed;
        }, set: function(e3) {
          this._readableState && (this._readableState.destroyed = e3);
        } }), A.prototype.destroy = m.destroy, A.prototype._undestroy = m.undestroy, A.prototype._destroy = function(e3, t3) {
          t3(e3);
        }, A.prototype.push = function(e3, t3) {
          var r3, n3 = this._readableState;
          return n3.objectMode ? r3 = true : "string" == typeof e3 && ((t3 = t3 || n3.defaultEncoding) !== n3.encoding && (e3 = c.from(e3, t3), t3 = ""), r3 = true), S(this, e3, t3, false, r3);
        }, A.prototype.unshift = function(e3) {
          return S(this, e3, null, true, false);
        }, A.prototype.isPaused = function() {
          return false === this._readableState.flowing;
        }, A.prototype.setEncoding = function(e3) {
          h || (h = r2(2553).s);
          var t3 = new h(e3);
          this._readableState.decoder = t3, this._readableState.encoding = this._readableState.decoder.encoding;
          for (var n3 = this._readableState.buffer.head, s2 = ""; null !== n3; )
            s2 += t3.write(n3.data), n3 = n3.next;
          return this._readableState.buffer.clear(), "" !== s2 && this._readableState.buffer.push(s2), this._readableState.length = s2.length, this;
        };
        var R = 1073741824;
        function N(e3, t3) {
          return e3 <= 0 || 0 === t3.length && t3.ended ? 0 : t3.objectMode ? 1 : e3 != e3 ? t3.flowing && t3.length ? t3.buffer.head.data.length : t3.length : (e3 > t3.highWaterMark && (t3.highWaterMark = function(e4) {
            return e4 >= R ? e4 = R : (e4--, e4 |= e4 >>> 1, e4 |= e4 >>> 2, e4 |= e4 >>> 4, e4 |= e4 >>> 8, e4 |= e4 >>> 16, e4++), e4;
          }(e3)), e3 <= t3.length ? e3 : t3.ended ? t3.length : (t3.needReadable = true, 0));
        }
        function C(e3) {
          var t3 = e3._readableState;
          i("emitReadable", t3.needReadable, t3.emittedReadable), t3.needReadable = false, t3.emittedReadable || (i("emitReadable", t3.flowing), t3.emittedReadable = true, s.nextTick(O, e3));
        }
        function O(e3) {
          var t3 = e3._readableState;
          i("emitReadable_", t3.destroyed, t3.length, t3.ended), t3.destroyed || !t3.length && !t3.ended || (e3.emit("readable"), t3.emittedReadable = false), t3.needReadable = !t3.flowing && !t3.ended && t3.length <= t3.highWaterMark, B(e3);
        }
        function k(e3, t3) {
          t3.readingMore || (t3.readingMore = true, s.nextTick(L, e3, t3));
        }
        function L(e3, t3) {
          for (; !t3.reading && !t3.ended && (t3.length < t3.highWaterMark || t3.flowing && 0 === t3.length); ) {
            var r3 = t3.length;
            if (i("maybeReadMore read 0"), e3.read(0), r3 === t3.length)
              break;
          }
          t3.readingMore = false;
        }
        function D(e3) {
          var t3 = e3._readableState;
          t3.readableListening = e3.listenerCount("readable") > 0, t3.resumeScheduled && !t3.paused ? t3.flowing = true : e3.listenerCount("data") > 0 && e3.resume();
        }
        function P(e3) {
          i("readable nexttick read 0"), e3.read(0);
        }
        function F(e3, t3) {
          i("resume", t3.reading), t3.reading || e3.read(0), t3.resumeScheduled = false, e3.emit("resume"), B(e3), t3.flowing && !t3.reading && e3.read(0);
        }
        function B(e3) {
          var t3 = e3._readableState;
          for (i("flow", t3.flowing); t3.flowing && null !== e3.read(); )
            ;
        }
        function M(e3, t3) {
          return 0 === t3.length ? null : (t3.objectMode ? r3 = t3.buffer.shift() : !e3 || e3 >= t3.length ? (r3 = t3.decoder ? t3.buffer.join("") : 1 === t3.buffer.length ? t3.buffer.first() : t3.buffer.concat(t3.length), t3.buffer.clear()) : r3 = t3.buffer.consume(e3, t3.decoder), r3);
          var r3;
        }
        function U(e3) {
          var t3 = e3._readableState;
          i("endReadable", t3.endEmitted), t3.endEmitted || (t3.ended = true, s.nextTick(H, t3, e3));
        }
        function H(e3, t3) {
          if (i("endReadableNT", e3.endEmitted, e3.length), !e3.endEmitted && 0 === e3.length && (e3.endEmitted = true, t3.readable = false, t3.emit("end"), e3.autoDestroy)) {
            var r3 = t3._writableState;
            (!r3 || r3.autoDestroy && r3.finished) && t3.destroy();
          }
        }
        function z(e3, t3) {
          for (var r3 = 0, n3 = e3.length; r3 < n3; r3++)
            if (e3[r3] === t3)
              return r3;
          return -1;
        }
        A.prototype.read = function(e3) {
          i("read", e3), e3 = parseInt(e3, 10);
          var t3 = this._readableState, r3 = e3;
          if (0 !== e3 && (t3.emittedReadable = false), 0 === e3 && t3.needReadable && ((0 !== t3.highWaterMark ? t3.length >= t3.highWaterMark : t3.length > 0) || t3.ended))
            return i("read: emitReadable", t3.length, t3.ended), 0 === t3.length && t3.ended ? U(this) : C(this), null;
          if (0 === (e3 = N(e3, t3)) && t3.ended)
            return 0 === t3.length && U(this), null;
          var n3, s2 = t3.needReadable;
          return i("need readable", s2), (0 === t3.length || t3.length - e3 < t3.highWaterMark) && i("length less than watermark", s2 = true), t3.ended || t3.reading ? i("reading or ended", s2 = false) : s2 && (i("do read"), t3.reading = true, t3.sync = true, 0 === t3.length && (t3.needReadable = true), this._read(t3.highWaterMark), t3.sync = false, t3.reading || (e3 = N(r3, t3))), null === (n3 = e3 > 0 ? M(e3, t3) : null) ? (t3.needReadable = t3.length <= t3.highWaterMark, e3 = 0) : (t3.length -= e3, t3.awaitDrain = 0), 0 === t3.length && (t3.ended || (t3.needReadable = true), r3 !== e3 && t3.ended && U(this)), null !== n3 && this.emit("data", n3), n3;
        }, A.prototype._read = function(e3) {
          _(this, new x("_read()"));
        }, A.prototype.pipe = function(e3, t3) {
          var r3 = this, n3 = this._readableState;
          switch (n3.pipesCount) {
            case 0:
              n3.pipes = e3;
              break;
            case 1:
              n3.pipes = [n3.pipes, e3];
              break;
            default:
              n3.pipes.push(e3);
          }
          n3.pipesCount += 1, i("pipe count=%d opts=%j", n3.pipesCount, t3);
          var a2 = t3 && false === t3.end || e3 === s.stdout || e3 === s.stderr ? m2 : c2;
          function c2() {
            i("onend"), e3.end();
          }
          n3.endEmitted ? s.nextTick(a2) : r3.once("end", a2), e3.on("unpipe", function t4(s2, o2) {
            i("onunpipe"), s2 === r3 && o2 && false === o2.hasUnpiped && (o2.hasUnpiped = true, i("cleanup"), e3.removeListener("close", d2), e3.removeListener("finish", f2), e3.removeListener("drain", u2), e3.removeListener("error", p2), e3.removeListener("unpipe", t4), r3.removeListener("end", c2), r3.removeListener("end", m2), r3.removeListener("data", h2), l2 = true, !n3.awaitDrain || e3._writableState && !e3._writableState.needDrain || u2());
          });
          var u2 = function(e4) {
            return function() {
              var t4 = e4._readableState;
              i("pipeOnDrain", t4.awaitDrain), t4.awaitDrain && t4.awaitDrain--, 0 === t4.awaitDrain && o(e4, "data") && (t4.flowing = true, B(e4));
            };
          }(r3);
          e3.on("drain", u2);
          var l2 = false;
          function h2(t4) {
            i("ondata");
            var s2 = e3.write(t4);
            i("dest.write", s2), false === s2 && ((1 === n3.pipesCount && n3.pipes === e3 || n3.pipesCount > 1 && -1 !== z(n3.pipes, e3)) && !l2 && (i("false write response, pause", n3.awaitDrain), n3.awaitDrain++), r3.pause());
          }
          function p2(t4) {
            i("onerror", t4), m2(), e3.removeListener("error", p2), 0 === o(e3, "error") && _(e3, t4);
          }
          function d2() {
            e3.removeListener("finish", f2), m2();
          }
          function f2() {
            i("onfinish"), e3.removeListener("close", d2), m2();
          }
          function m2() {
            i("unpipe"), r3.unpipe(e3);
          }
          return r3.on("data", h2), function(e4, t4, r4) {
            if ("function" == typeof e4.prependListener)
              return e4.prependListener(t4, r4);
            e4._events && e4._events.error ? Array.isArray(e4._events.error) ? e4._events.error.unshift(r4) : e4._events.error = [r4, e4._events.error] : e4.on(t4, r4);
          }(e3, "error", p2), e3.once("close", d2), e3.once("finish", f2), e3.emit("pipe", r3), n3.flowing || (i("pipe resume"), r3.resume()), e3;
        }, A.prototype.unpipe = function(e3) {
          var t3 = this._readableState, r3 = { hasUnpiped: false };
          if (0 === t3.pipesCount)
            return this;
          if (1 === t3.pipesCount)
            return e3 && e3 !== t3.pipes || (e3 || (e3 = t3.pipes), t3.pipes = null, t3.pipesCount = 0, t3.flowing = false, e3 && e3.emit("unpipe", this, r3)), this;
          if (!e3) {
            var n3 = t3.pipes, s2 = t3.pipesCount;
            t3.pipes = null, t3.pipesCount = 0, t3.flowing = false;
            for (var i2 = 0; i2 < s2; i2++)
              n3[i2].emit("unpipe", this, { hasUnpiped: false });
            return this;
          }
          var o2 = z(t3.pipes, e3);
          return -1 === o2 || (t3.pipes.splice(o2, 1), t3.pipesCount -= 1, 1 === t3.pipesCount && (t3.pipes = t3.pipes[0]), e3.emit("unpipe", this, r3)), this;
        }, A.prototype.on = function(e3, t3) {
          var r3 = a.prototype.on.call(this, e3, t3), n3 = this._readableState;
          return "data" === e3 ? (n3.readableListening = this.listenerCount("readable") > 0, false !== n3.flowing && this.resume()) : "readable" === e3 && (n3.endEmitted || n3.readableListening || (n3.readableListening = n3.needReadable = true, n3.flowing = false, n3.emittedReadable = false, i("on readable", n3.length, n3.reading), n3.length ? C(this) : n3.reading || s.nextTick(P, this))), r3;
        }, A.prototype.addListener = A.prototype.on, A.prototype.removeListener = function(e3, t3) {
          var r3 = a.prototype.removeListener.call(this, e3, t3);
          return "readable" === e3 && s.nextTick(D, this), r3;
        }, A.prototype.removeAllListeners = function(e3) {
          var t3 = a.prototype.removeAllListeners.apply(this, arguments);
          return "readable" !== e3 && void 0 !== e3 || s.nextTick(D, this), t3;
        }, A.prototype.resume = function() {
          var e3 = this._readableState;
          return e3.flowing || (i("resume"), e3.flowing = !e3.readableListening, function(e4, t3) {
            t3.resumeScheduled || (t3.resumeScheduled = true, s.nextTick(F, e4, t3));
          }(this, e3)), e3.paused = false, this;
        }, A.prototype.pause = function() {
          return i("call pause flowing=%j", this._readableState.flowing), false !== this._readableState.flowing && (i("pause"), this._readableState.flowing = false, this.emit("pause")), this._readableState.paused = true, this;
        }, A.prototype.wrap = function(e3) {
          var t3 = this, r3 = this._readableState, n3 = false;
          for (var s2 in e3.on("end", function() {
            if (i("wrapped end"), r3.decoder && !r3.ended) {
              var e4 = r3.decoder.end();
              e4 && e4.length && t3.push(e4);
            }
            t3.push(null);
          }), e3.on("data", function(s3) {
            i("wrapped data"), r3.decoder && (s3 = r3.decoder.write(s3)), r3.objectMode && null == s3 || (r3.objectMode || s3 && s3.length) && (t3.push(s3) || (n3 = true, e3.pause()));
          }), e3)
            void 0 === this[s2] && "function" == typeof e3[s2] && (this[s2] = function(t4) {
              return function() {
                return e3[t4].apply(e3, arguments);
              };
            }(s2));
          for (var o2 = 0; o2 < E.length; o2++)
            e3.on(E[o2], this.emit.bind(this, E[o2]));
          return this._read = function(t4) {
            i("wrapped _read", t4), n3 && (n3 = false, e3.resume());
          }, this;
        }, "function" == typeof Symbol && (A.prototype[Symbol.asyncIterator] = function() {
          return void 0 === p && (p = r2(828)), p(this);
        }), Object.defineProperty(A.prototype, "readableHighWaterMark", { enumerable: false, get: function() {
          return this._readableState.highWaterMark;
        } }), Object.defineProperty(A.prototype, "readableBuffer", { enumerable: false, get: function() {
          return this._readableState && this._readableState.buffer;
        } }), Object.defineProperty(A.prototype, "readableFlowing", { enumerable: false, get: function() {
          return this._readableState.flowing;
        }, set: function(e3) {
          this._readableState && (this._readableState.flowing = e3);
        } }), A._fromList = M, Object.defineProperty(A.prototype, "readableLength", { enumerable: false, get: function() {
          return this._readableState.length;
        } }), "function" == typeof Symbol && (A.from = function(e3, t3) {
          return void 0 === d && (d = r2(1265)), d(A, e3, t3);
        });
      }, 4473: (e2, t2, r2) => {
        "use strict";
        e2.exports = l;
        var n2 = r2(8106).q, s = n2.ERR_METHOD_NOT_IMPLEMENTED, i = n2.ERR_MULTIPLE_CALLBACK, o = n2.ERR_TRANSFORM_ALREADY_TRANSFORMING, a = n2.ERR_TRANSFORM_WITH_LENGTH_0, c = r2(8656);
        function u(e3, t3) {
          var r3 = this._transformState;
          r3.transforming = false;
          var n3 = r3.writecb;
          if (null === n3)
            return this.emit("error", new i());
          r3.writechunk = null, r3.writecb = null, null != t3 && this.push(t3), n3(e3);
          var s2 = this._readableState;
          s2.reading = false, (s2.needReadable || s2.length < s2.highWaterMark) && this._read(s2.highWaterMark);
        }
        function l(e3) {
          if (!(this instanceof l))
            return new l(e3);
          c.call(this, e3), this._transformState = { afterTransform: u.bind(this), needTransform: false, transforming: false, writecb: null, writechunk: null, writeencoding: null }, this._readableState.needReadable = true, this._readableState.sync = false, e3 && ("function" == typeof e3.transform && (this._transform = e3.transform), "function" == typeof e3.flush && (this._flush = e3.flush)), this.on("prefinish", h);
        }
        function h() {
          var e3 = this;
          "function" != typeof this._flush || this._readableState.destroyed ? p(this, null, null) : this._flush(function(t3, r3) {
            p(e3, t3, r3);
          });
        }
        function p(e3, t3, r3) {
          if (t3)
            return e3.emit("error", t3);
          if (null != r3 && e3.push(r3), e3._writableState.length)
            throw new a();
          if (e3._transformState.transforming)
            throw new o();
          return e3.push(null);
        }
        r2(5717)(l, c), l.prototype.push = function(e3, t3) {
          return this._transformState.needTransform = false, c.prototype.push.call(this, e3, t3);
        }, l.prototype._transform = function(e3, t3, r3) {
          r3(new s("_transform()"));
        }, l.prototype._write = function(e3, t3, r3) {
          var n3 = this._transformState;
          if (n3.writecb = r3, n3.writechunk = e3, n3.writeencoding = t3, !n3.transforming) {
            var s2 = this._readableState;
            (n3.needTransform || s2.needReadable || s2.length < s2.highWaterMark) && this._read(s2.highWaterMark);
          }
        }, l.prototype._read = function(e3) {
          var t3 = this._transformState;
          null === t3.writechunk || t3.transforming ? t3.needTransform = true : (t3.transforming = true, this._transform(t3.writechunk, t3.writeencoding, t3.afterTransform));
        }, l.prototype._destroy = function(e3, t3) {
          c.prototype._destroy.call(this, e3, function(e4) {
            t3(e4);
          });
        };
      }, 323: (e2, t2, r2) => {
        "use strict";
        var n2, s = r2(4155);
        function i(e3) {
          var t3 = this;
          this.next = null, this.entry = null, this.finish = function() {
            !function(e4, t4, r3) {
              var n3 = e4.entry;
              for (e4.entry = null; n3; ) {
                var s2 = n3.callback;
                t4.pendingcb--, s2(void 0), n3 = n3.next;
              }
              t4.corkedRequestsFree.next = e4;
            }(t3, e3);
          };
        }
        e2.exports = A, A.WritableState = T;
        var o, a = { deprecate: r2(4927) }, c = r2(3194), u = r2(8764).Buffer, l = r2.g.Uint8Array || function() {
        }, h = r2(1029), p = r2(94).getHighWaterMark, d = r2(8106).q, f = d.ERR_INVALID_ARG_TYPE, m = d.ERR_METHOD_NOT_IMPLEMENTED, w = d.ERR_MULTIPLE_CALLBACK, g = d.ERR_STREAM_CANNOT_PIPE, y = d.ERR_STREAM_DESTROYED, b = d.ERR_STREAM_NULL_VALUES, x = d.ERR_STREAM_WRITE_AFTER_END, v = d.ERR_UNKNOWN_ENCODING, _ = h.errorOrDestroy;
        function E() {
        }
        function T(e3, t3, o2) {
          n2 = n2 || r2(8656), e3 = e3 || {}, "boolean" != typeof o2 && (o2 = t3 instanceof n2), this.objectMode = !!e3.objectMode, o2 && (this.objectMode = this.objectMode || !!e3.writableObjectMode), this.highWaterMark = p(this, e3, "writableHighWaterMark", o2), this.finalCalled = false, this.needDrain = false, this.ending = false, this.ended = false, this.finished = false, this.destroyed = false;
          var a2 = false === e3.decodeStrings;
          this.decodeStrings = !a2, this.defaultEncoding = e3.defaultEncoding || "utf8", this.length = 0, this.writing = false, this.corked = 0, this.sync = true, this.bufferProcessing = false, this.onwrite = function(e4) {
            !function(e5, t4) {
              var r3 = e5._writableState, n3 = r3.sync, i2 = r3.writecb;
              if ("function" != typeof i2)
                throw new w();
              if (function(e6) {
                e6.writing = false, e6.writecb = null, e6.length -= e6.writelen, e6.writelen = 0;
              }(r3), t4)
                !function(e6, t5, r4, n4, i3) {
                  --t5.pendingcb, r4 ? (s.nextTick(i3, n4), s.nextTick(O, e6, t5), e6._writableState.errorEmitted = true, _(e6, n4)) : (i3(n4), e6._writableState.errorEmitted = true, _(e6, n4), O(e6, t5));
                }(e5, r3, n3, t4, i2);
              else {
                var o3 = N(r3) || e5.destroyed;
                o3 || r3.corked || r3.bufferProcessing || !r3.bufferedRequest || R(e5, r3), n3 ? s.nextTick(I, e5, r3, o3, i2) : I(e5, r3, o3, i2);
              }
            }(t3, e4);
          }, this.writecb = null, this.writelen = 0, this.bufferedRequest = null, this.lastBufferedRequest = null, this.pendingcb = 0, this.prefinished = false, this.errorEmitted = false, this.emitClose = false !== e3.emitClose, this.autoDestroy = !!e3.autoDestroy, this.bufferedRequestCount = 0, this.corkedRequestsFree = new i(this);
        }
        function A(e3) {
          var t3 = this instanceof (n2 = n2 || r2(8656));
          if (!t3 && !o.call(A, this))
            return new A(e3);
          this._writableState = new T(e3, this, t3), this.writable = true, e3 && ("function" == typeof e3.write && (this._write = e3.write), "function" == typeof e3.writev && (this._writev = e3.writev), "function" == typeof e3.destroy && (this._destroy = e3.destroy), "function" == typeof e3.final && (this._final = e3.final)), c.call(this);
        }
        function S(e3, t3, r3, n3, s2, i2, o2) {
          t3.writelen = n3, t3.writecb = o2, t3.writing = true, t3.sync = true, t3.destroyed ? t3.onwrite(new y("write")) : r3 ? e3._writev(s2, t3.onwrite) : e3._write(s2, i2, t3.onwrite), t3.sync = false;
        }
        function I(e3, t3, r3, n3) {
          r3 || function(e4, t4) {
            0 === t4.length && t4.needDrain && (t4.needDrain = false, e4.emit("drain"));
          }(e3, t3), t3.pendingcb--, n3(), O(e3, t3);
        }
        function R(e3, t3) {
          t3.bufferProcessing = true;
          var r3 = t3.bufferedRequest;
          if (e3._writev && r3 && r3.next) {
            var n3 = t3.bufferedRequestCount, s2 = new Array(n3), o2 = t3.corkedRequestsFree;
            o2.entry = r3;
            for (var a2 = 0, c2 = true; r3; )
              s2[a2] = r3, r3.isBuf || (c2 = false), r3 = r3.next, a2 += 1;
            s2.allBuffers = c2, S(e3, t3, true, t3.length, s2, "", o2.finish), t3.pendingcb++, t3.lastBufferedRequest = null, o2.next ? (t3.corkedRequestsFree = o2.next, o2.next = null) : t3.corkedRequestsFree = new i(t3), t3.bufferedRequestCount = 0;
          } else {
            for (; r3; ) {
              var u2 = r3.chunk, l2 = r3.encoding, h2 = r3.callback;
              if (S(e3, t3, false, t3.objectMode ? 1 : u2.length, u2, l2, h2), r3 = r3.next, t3.bufferedRequestCount--, t3.writing)
                break;
            }
            null === r3 && (t3.lastBufferedRequest = null);
          }
          t3.bufferedRequest = r3, t3.bufferProcessing = false;
        }
        function N(e3) {
          return e3.ending && 0 === e3.length && null === e3.bufferedRequest && !e3.finished && !e3.writing;
        }
        function C(e3, t3) {
          e3._final(function(r3) {
            t3.pendingcb--, r3 && _(e3, r3), t3.prefinished = true, e3.emit("prefinish"), O(e3, t3);
          });
        }
        function O(e3, t3) {
          var r3 = N(t3);
          if (r3 && (function(e4, t4) {
            t4.prefinished || t4.finalCalled || ("function" != typeof e4._final || t4.destroyed ? (t4.prefinished = true, e4.emit("prefinish")) : (t4.pendingcb++, t4.finalCalled = true, s.nextTick(C, e4, t4)));
          }(e3, t3), 0 === t3.pendingcb && (t3.finished = true, e3.emit("finish"), t3.autoDestroy))) {
            var n3 = e3._readableState;
            (!n3 || n3.autoDestroy && n3.endEmitted) && e3.destroy();
          }
          return r3;
        }
        r2(5717)(A, c), T.prototype.getBuffer = function() {
          for (var e3 = this.bufferedRequest, t3 = []; e3; )
            t3.push(e3), e3 = e3.next;
          return t3;
        }, function() {
          try {
            Object.defineProperty(T.prototype, "buffer", { get: a.deprecate(function() {
              return this.getBuffer();
            }, "_writableState.buffer is deprecated. Use _writableState.getBuffer instead.", "DEP0003") });
          } catch (e3) {
          }
        }(), "function" == typeof Symbol && Symbol.hasInstance && "function" == typeof Function.prototype[Symbol.hasInstance] ? (o = Function.prototype[Symbol.hasInstance], Object.defineProperty(A, Symbol.hasInstance, { value: function(e3) {
          return !!o.call(this, e3) || this === A && e3 && e3._writableState instanceof T;
        } })) : o = function(e3) {
          return e3 instanceof this;
        }, A.prototype.pipe = function() {
          _(this, new g());
        }, A.prototype.write = function(e3, t3, r3) {
          var n3, i2 = this._writableState, o2 = false, a2 = !i2.objectMode && (n3 = e3, u.isBuffer(n3) || n3 instanceof l);
          return a2 && !u.isBuffer(e3) && (e3 = function(e4) {
            return u.from(e4);
          }(e3)), "function" == typeof t3 && (r3 = t3, t3 = null), a2 ? t3 = "buffer" : t3 || (t3 = i2.defaultEncoding), "function" != typeof r3 && (r3 = E), i2.ending ? function(e4, t4) {
            var r4 = new x();
            _(e4, r4), s.nextTick(t4, r4);
          }(this, r3) : (a2 || function(e4, t4, r4, n4) {
            var i3;
            return null === r4 ? i3 = new b() : "string" == typeof r4 || t4.objectMode || (i3 = new f("chunk", ["string", "Buffer"], r4)), !i3 || (_(e4, i3), s.nextTick(n4, i3), false);
          }(this, i2, e3, r3)) && (i2.pendingcb++, o2 = function(e4, t4, r4, n4, s2, i3) {
            if (!r4) {
              var o3 = function(e5, t5, r5) {
                return e5.objectMode || false === e5.decodeStrings || "string" != typeof t5 || (t5 = u.from(t5, r5)), t5;
              }(t4, n4, s2);
              n4 !== o3 && (r4 = true, s2 = "buffer", n4 = o3);
            }
            var a3 = t4.objectMode ? 1 : n4.length;
            t4.length += a3;
            var c2 = t4.length < t4.highWaterMark;
            if (c2 || (t4.needDrain = true), t4.writing || t4.corked) {
              var l2 = t4.lastBufferedRequest;
              t4.lastBufferedRequest = { chunk: n4, encoding: s2, isBuf: r4, callback: i3, next: null }, l2 ? l2.next = t4.lastBufferedRequest : t4.bufferedRequest = t4.lastBufferedRequest, t4.bufferedRequestCount += 1;
            } else
              S(e4, t4, false, a3, n4, s2, i3);
            return c2;
          }(this, i2, a2, e3, t3, r3)), o2;
        }, A.prototype.cork = function() {
          this._writableState.corked++;
        }, A.prototype.uncork = function() {
          var e3 = this._writableState;
          e3.corked && (e3.corked--, e3.writing || e3.corked || e3.bufferProcessing || !e3.bufferedRequest || R(this, e3));
        }, A.prototype.setDefaultEncoding = function(e3) {
          if ("string" == typeof e3 && (e3 = e3.toLowerCase()), !(["hex", "utf8", "utf-8", "ascii", "binary", "base64", "ucs2", "ucs-2", "utf16le", "utf-16le", "raw"].indexOf((e3 + "").toLowerCase()) > -1))
            throw new v(e3);
          return this._writableState.defaultEncoding = e3, this;
        }, Object.defineProperty(A.prototype, "writableBuffer", { enumerable: false, get: function() {
          return this._writableState && this._writableState.getBuffer();
        } }), Object.defineProperty(A.prototype, "writableHighWaterMark", { enumerable: false, get: function() {
          return this._writableState.highWaterMark;
        } }), A.prototype._write = function(e3, t3, r3) {
          r3(new m("_write()"));
        }, A.prototype._writev = null, A.prototype.end = function(e3, t3, r3) {
          var n3 = this._writableState;
          return "function" == typeof e3 ? (r3 = e3, e3 = null, t3 = null) : "function" == typeof t3 && (r3 = t3, t3 = null), null != e3 && this.write(e3, t3), n3.corked && (n3.corked = 1, this.uncork()), n3.ending || function(e4, t4, r4) {
            t4.ending = true, O(e4, t4), r4 && (t4.finished ? s.nextTick(r4) : e4.once("finish", r4)), t4.ended = true, e4.writable = false;
          }(this, n3, r3), this;
        }, Object.defineProperty(A.prototype, "writableLength", { enumerable: false, get: function() {
          return this._writableState.length;
        } }), Object.defineProperty(A.prototype, "destroyed", { enumerable: false, get: function() {
          return void 0 !== this._writableState && this._writableState.destroyed;
        }, set: function(e3) {
          this._writableState && (this._writableState.destroyed = e3);
        } }), A.prototype.destroy = h.destroy, A.prototype._undestroy = h.undestroy, A.prototype._destroy = function(e3, t3) {
          t3(e3);
        };
      }, 828: (e2, t2, r2) => {
        "use strict";
        var n2, s = r2(4155);
        function i(e3, t3, r3) {
          return t3 in e3 ? Object.defineProperty(e3, t3, { value: r3, enumerable: true, configurable: true, writable: true }) : e3[t3] = r3, e3;
        }
        var o = r2(1086), a = Symbol("lastResolve"), c = Symbol("lastReject"), u = Symbol("error"), l = Symbol("ended"), h = Symbol("lastPromise"), p = Symbol("handlePromise"), d = Symbol("stream");
        function f(e3, t3) {
          return { value: e3, done: t3 };
        }
        function m(e3) {
          var t3 = e3[a];
          if (null !== t3) {
            var r3 = e3[d].read();
            null !== r3 && (e3[h] = null, e3[a] = null, e3[c] = null, t3(f(r3, false)));
          }
        }
        function w(e3) {
          s.nextTick(m, e3);
        }
        var g = Object.getPrototypeOf(function() {
        }), y = Object.setPrototypeOf((i(n2 = { get stream() {
          return this[d];
        }, next: function() {
          var e3 = this, t3 = this[u];
          if (null !== t3)
            return Promise.reject(t3);
          if (this[l])
            return Promise.resolve(f(void 0, true));
          if (this[d].destroyed)
            return new Promise(function(t4, r4) {
              s.nextTick(function() {
                e3[u] ? r4(e3[u]) : t4(f(void 0, true));
              });
            });
          var r3, n3 = this[h];
          if (n3)
            r3 = new Promise(function(e4, t4) {
              return function(r4, n4) {
                e4.then(function() {
                  t4[l] ? r4(f(void 0, true)) : t4[p](r4, n4);
                }, n4);
              };
            }(n3, this));
          else {
            var i2 = this[d].read();
            if (null !== i2)
              return Promise.resolve(f(i2, false));
            r3 = new Promise(this[p]);
          }
          return this[h] = r3, r3;
        } }, Symbol.asyncIterator, function() {
          return this;
        }), i(n2, "return", function() {
          var e3 = this;
          return new Promise(function(t3, r3) {
            e3[d].destroy(null, function(e4) {
              e4 ? r3(e4) : t3(f(void 0, true));
            });
          });
        }), n2), g);
        e2.exports = function(e3) {
          var t3, r3 = Object.create(y, (i(t3 = {}, d, { value: e3, writable: true }), i(t3, a, { value: null, writable: true }), i(t3, c, { value: null, writable: true }), i(t3, u, { value: null, writable: true }), i(t3, l, { value: e3._readableState.endEmitted, writable: true }), i(t3, p, { value: function(e4, t4) {
            var n3 = r3[d].read();
            n3 ? (r3[h] = null, r3[a] = null, r3[c] = null, e4(f(n3, false))) : (r3[a] = e4, r3[c] = t4);
          }, writable: true }), t3));
          return r3[h] = null, o(e3, function(e4) {
            if (e4 && "ERR_STREAM_PREMATURE_CLOSE" !== e4.code) {
              var t4 = r3[c];
              return null !== t4 && (r3[h] = null, r3[a] = null, r3[c] = null, t4(e4)), void (r3[u] = e4);
            }
            var n3 = r3[a];
            null !== n3 && (r3[h] = null, r3[a] = null, r3[c] = null, n3(f(void 0, true))), r3[l] = true;
          }), e3.on("readable", w.bind(null, r3)), r3;
        };
      }, 9686: (e2, t2, r2) => {
        "use strict";
        function n2(e3, t3) {
          var r3 = Object.keys(e3);
          if (Object.getOwnPropertySymbols) {
            var n3 = Object.getOwnPropertySymbols(e3);
            t3 && (n3 = n3.filter(function(t4) {
              return Object.getOwnPropertyDescriptor(e3, t4).enumerable;
            })), r3.push.apply(r3, n3);
          }
          return r3;
        }
        function s(e3, t3, r3) {
          return t3 in e3 ? Object.defineProperty(e3, t3, { value: r3, enumerable: true, configurable: true, writable: true }) : e3[t3] = r3, e3;
        }
        function i(e3, t3) {
          for (var r3 = 0; r3 < t3.length; r3++) {
            var n3 = t3[r3];
            n3.enumerable = n3.enumerable || false, n3.configurable = true, "value" in n3 && (n3.writable = true), Object.defineProperty(e3, n3.key, n3);
          }
        }
        var o = r2(8764).Buffer, a = r2(5575).inspect, c = a && a.custom || "inspect";
        e2.exports = function() {
          function e3() {
            !function(e4, t4) {
              if (!(e4 instanceof t4))
                throw new TypeError("Cannot call a class as a function");
            }(this, e3), this.head = null, this.tail = null, this.length = 0;
          }
          var t3, r3;
          return t3 = e3, r3 = [{ key: "push", value: function(e4) {
            var t4 = { data: e4, next: null };
            this.length > 0 ? this.tail.next = t4 : this.head = t4, this.tail = t4, ++this.length;
          } }, { key: "unshift", value: function(e4) {
            var t4 = { data: e4, next: this.head };
            0 === this.length && (this.tail = t4), this.head = t4, ++this.length;
          } }, { key: "shift", value: function() {
            if (0 !== this.length) {
              var e4 = this.head.data;
              return 1 === this.length ? this.head = this.tail = null : this.head = this.head.next, --this.length, e4;
            }
          } }, { key: "clear", value: function() {
            this.head = this.tail = null, this.length = 0;
          } }, { key: "join", value: function(e4) {
            if (0 === this.length)
              return "";
            for (var t4 = this.head, r4 = "" + t4.data; t4 = t4.next; )
              r4 += e4 + t4.data;
            return r4;
          } }, { key: "concat", value: function(e4) {
            if (0 === this.length)
              return o.alloc(0);
            for (var t4, r4, n3, s2 = o.allocUnsafe(e4 >>> 0), i2 = this.head, a2 = 0; i2; )
              t4 = i2.data, r4 = s2, n3 = a2, o.prototype.copy.call(t4, r4, n3), a2 += i2.data.length, i2 = i2.next;
            return s2;
          } }, { key: "consume", value: function(e4, t4) {
            var r4;
            return e4 < this.head.data.length ? (r4 = this.head.data.slice(0, e4), this.head.data = this.head.data.slice(e4)) : r4 = e4 === this.head.data.length ? this.shift() : t4 ? this._getString(e4) : this._getBuffer(e4), r4;
          } }, { key: "first", value: function() {
            return this.head.data;
          } }, { key: "_getString", value: function(e4) {
            var t4 = this.head, r4 = 1, n3 = t4.data;
            for (e4 -= n3.length; t4 = t4.next; ) {
              var s2 = t4.data, i2 = e4 > s2.length ? s2.length : e4;
              if (i2 === s2.length ? n3 += s2 : n3 += s2.slice(0, e4), 0 == (e4 -= i2)) {
                i2 === s2.length ? (++r4, t4.next ? this.head = t4.next : this.head = this.tail = null) : (this.head = t4, t4.data = s2.slice(i2));
                break;
              }
              ++r4;
            }
            return this.length -= r4, n3;
          } }, { key: "_getBuffer", value: function(e4) {
            var t4 = o.allocUnsafe(e4), r4 = this.head, n3 = 1;
            for (r4.data.copy(t4), e4 -= r4.data.length; r4 = r4.next; ) {
              var s2 = r4.data, i2 = e4 > s2.length ? s2.length : e4;
              if (s2.copy(t4, t4.length - e4, 0, i2), 0 == (e4 -= i2)) {
                i2 === s2.length ? (++n3, r4.next ? this.head = r4.next : this.head = this.tail = null) : (this.head = r4, r4.data = s2.slice(i2));
                break;
              }
              ++n3;
            }
            return this.length -= n3, t4;
          } }, { key: c, value: function(e4, t4) {
            return a(this, function(e5) {
              for (var t5 = 1; t5 < arguments.length; t5++) {
                var r4 = null != arguments[t5] ? arguments[t5] : {};
                t5 % 2 ? n2(Object(r4), true).forEach(function(t6) {
                  s(e5, t6, r4[t6]);
                }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e5, Object.getOwnPropertyDescriptors(r4)) : n2(Object(r4)).forEach(function(t6) {
                  Object.defineProperty(e5, t6, Object.getOwnPropertyDescriptor(r4, t6));
                });
              }
              return e5;
            }({}, t4, { depth: 0, customInspect: false }));
          } }], r3 && i(t3.prototype, r3), e3;
        }();
      }, 1029: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(4155);
        function s(e3, t3) {
          o(e3, t3), i(e3);
        }
        function i(e3) {
          e3._writableState && !e3._writableState.emitClose || e3._readableState && !e3._readableState.emitClose || e3.emit("close");
        }
        function o(e3, t3) {
          e3.emit("error", t3);
        }
        e2.exports = { destroy: function(e3, t3) {
          var r3 = this, a = this._readableState && this._readableState.destroyed, c = this._writableState && this._writableState.destroyed;
          return a || c ? (t3 ? t3(e3) : e3 && (this._writableState ? this._writableState.errorEmitted || (this._writableState.errorEmitted = true, n2.nextTick(o, this, e3)) : n2.nextTick(o, this, e3)), this) : (this._readableState && (this._readableState.destroyed = true), this._writableState && (this._writableState.destroyed = true), this._destroy(e3 || null, function(e4) {
            !t3 && e4 ? r3._writableState ? r3._writableState.errorEmitted ? n2.nextTick(i, r3) : (r3._writableState.errorEmitted = true, n2.nextTick(s, r3, e4)) : n2.nextTick(s, r3, e4) : t3 ? (n2.nextTick(i, r3), t3(e4)) : n2.nextTick(i, r3);
          }), this);
        }, undestroy: function() {
          this._readableState && (this._readableState.destroyed = false, this._readableState.reading = false, this._readableState.ended = false, this._readableState.endEmitted = false), this._writableState && (this._writableState.destroyed = false, this._writableState.ended = false, this._writableState.ending = false, this._writableState.finalCalled = false, this._writableState.prefinished = false, this._writableState.finished = false, this._writableState.errorEmitted = false);
        }, errorOrDestroy: function(e3, t3) {
          var r3 = e3._readableState, n3 = e3._writableState;
          r3 && r3.autoDestroy || n3 && n3.autoDestroy ? e3.destroy(t3) : e3.emit("error", t3);
        } };
      }, 1086: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(8106).q.ERR_STREAM_PREMATURE_CLOSE;
        function s() {
        }
        e2.exports = function e3(t3, r3, i) {
          if ("function" == typeof r3)
            return e3(t3, null, r3);
          r3 || (r3 = {}), i = function(e4) {
            var t4 = false;
            return function() {
              if (!t4) {
                t4 = true;
                for (var r4 = arguments.length, n3 = new Array(r4), s2 = 0; s2 < r4; s2++)
                  n3[s2] = arguments[s2];
                e4.apply(this, n3);
              }
            };
          }(i || s);
          var o = r3.readable || false !== r3.readable && t3.readable, a = r3.writable || false !== r3.writable && t3.writable, c = function() {
            t3.writable || l();
          }, u = t3._writableState && t3._writableState.finished, l = function() {
            a = false, u = true, o || i.call(t3);
          }, h = t3._readableState && t3._readableState.endEmitted, p = function() {
            o = false, h = true, a || i.call(t3);
          }, d = function(e4) {
            i.call(t3, e4);
          }, f = function() {
            var e4;
            return o && !h ? (t3._readableState && t3._readableState.ended || (e4 = new n2()), i.call(t3, e4)) : a && !u ? (t3._writableState && t3._writableState.ended || (e4 = new n2()), i.call(t3, e4)) : void 0;
          }, m = function() {
            t3.req.on("finish", l);
          };
          return function(e4) {
            return e4.setHeader && "function" == typeof e4.abort;
          }(t3) ? (t3.on("complete", l), t3.on("abort", f), t3.req ? m() : t3.on("request", m)) : a && !t3._writableState && (t3.on("end", c), t3.on("close", c)), t3.on("end", p), t3.on("finish", l), false !== r3.error && t3.on("error", d), t3.on("close", f), function() {
            t3.removeListener("complete", l), t3.removeListener("abort", f), t3.removeListener("request", m), t3.req && t3.req.removeListener("finish", l), t3.removeListener("end", c), t3.removeListener("close", c), t3.removeListener("finish", l), t3.removeListener("end", p), t3.removeListener("error", d), t3.removeListener("close", f);
          };
        };
      }, 1265: (e2) => {
        e2.exports = function() {
          throw new Error("Readable.from is not available in the browser");
        };
      }, 6472: (e2, t2, r2) => {
        "use strict";
        var n2, s = r2(8106).q, i = s.ERR_MISSING_ARGS, o = s.ERR_STREAM_DESTROYED;
        function a(e3) {
          if (e3)
            throw e3;
        }
        function c(e3, t3, s2, i2) {
          i2 = function(e4) {
            var t4 = false;
            return function() {
              t4 || (t4 = true, e4.apply(void 0, arguments));
            };
          }(i2);
          var a2 = false;
          e3.on("close", function() {
            a2 = true;
          }), void 0 === n2 && (n2 = r2(1086)), n2(e3, { readable: t3, writable: s2 }, function(e4) {
            if (e4)
              return i2(e4);
            a2 = true, i2();
          });
          var c2 = false;
          return function(t4) {
            if (!a2 && !c2)
              return c2 = true, function(e4) {
                return e4.setHeader && "function" == typeof e4.abort;
              }(e3) ? e3.abort() : "function" == typeof e3.destroy ? e3.destroy() : void i2(t4 || new o("pipe"));
          };
        }
        function u(e3) {
          e3();
        }
        function l(e3, t3) {
          return e3.pipe(t3);
        }
        function h(e3) {
          return e3.length ? "function" != typeof e3[e3.length - 1] ? a : e3.pop() : a;
        }
        e2.exports = function() {
          for (var e3 = arguments.length, t3 = new Array(e3), r3 = 0; r3 < e3; r3++)
            t3[r3] = arguments[r3];
          var n3, s2 = h(t3);
          if (Array.isArray(t3[0]) && (t3 = t3[0]), t3.length < 2)
            throw new i("streams");
          var o2 = t3.map(function(e4, r4) {
            var i2 = r4 < t3.length - 1;
            return c(e4, i2, r4 > 0, function(e5) {
              n3 || (n3 = e5), e5 && o2.forEach(u), i2 || (o2.forEach(u), s2(n3));
            });
          });
          return t3.reduce(l);
        };
      }, 94: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(8106).q.ERR_INVALID_OPT_VALUE;
        e2.exports = { getHighWaterMark: function(e3, t3, r3, s) {
          var i = function(e4, t4, r4) {
            return null != e4.highWaterMark ? e4.highWaterMark : t4 ? e4[r4] : null;
          }(t3, s, r3);
          if (null != i) {
            if (!isFinite(i) || Math.floor(i) !== i || i < 0)
              throw new n2(s ? r3 : "highWaterMark", i);
            return Math.floor(i);
          }
          return e3.objectMode ? 16 : 16384;
        } };
      }, 3194: (e2, t2, r2) => {
        e2.exports = r2(7187).EventEmitter;
      }, 2553: (e2, t2, r2) => {
        "use strict";
        var n2 = r2(9509).Buffer, s = n2.isEncoding || function(e3) {
          switch ((e3 = "" + e3) && e3.toLowerCase()) {
            case "hex":
            case "utf8":
            case "utf-8":
            case "ascii":
            case "binary":
            case "base64":
            case "ucs2":
            case "ucs-2":
            case "utf16le":
            case "utf-16le":
            case "raw":
              return true;
            default:
              return false;
          }
        };
        function i(e3) {
          var t3;
          switch (this.encoding = function(e4) {
            var t4 = function(e5) {
              if (!e5)
                return "utf8";
              for (var t5; ; )
                switch (e5) {
                  case "utf8":
                  case "utf-8":
                    return "utf8";
                  case "ucs2":
                  case "ucs-2":
                  case "utf16le":
                  case "utf-16le":
                    return "utf16le";
                  case "latin1":
                  case "binary":
                    return "latin1";
                  case "base64":
                  case "ascii":
                  case "hex":
                    return e5;
                  default:
                    if (t5)
                      return;
                    e5 = ("" + e5).toLowerCase(), t5 = true;
                }
            }(e4);
            if ("string" != typeof t4 && (n2.isEncoding === s || !s(e4)))
              throw new Error("Unknown encoding: " + e4);
            return t4 || e4;
          }(e3), this.encoding) {
            case "utf16le":
              this.text = c, this.end = u, t3 = 4;
              break;
            case "utf8":
              this.fillLast = a, t3 = 4;
              break;
            case "base64":
              this.text = l, this.end = h, t3 = 3;
              break;
            default:
              return this.write = p, void (this.end = d);
          }
          this.lastNeed = 0, this.lastTotal = 0, this.lastChar = n2.allocUnsafe(t3);
        }
        function o(e3) {
          return e3 <= 127 ? 0 : e3 >> 5 == 6 ? 2 : e3 >> 4 == 14 ? 3 : e3 >> 3 == 30 ? 4 : e3 >> 6 == 2 ? -1 : -2;
        }
        function a(e3) {
          var t3 = this.lastTotal - this.lastNeed, r3 = function(e4, t4, r4) {
            if (128 != (192 & t4[0]))
              return e4.lastNeed = 0, "\uFFFD";
            if (e4.lastNeed > 1 && t4.length > 1) {
              if (128 != (192 & t4[1]))
                return e4.lastNeed = 1, "\uFFFD";
              if (e4.lastNeed > 2 && t4.length > 2 && 128 != (192 & t4[2]))
                return e4.lastNeed = 2, "\uFFFD";
            }
          }(this, e3);
          return void 0 !== r3 ? r3 : this.lastNeed <= e3.length ? (e3.copy(this.lastChar, t3, 0, this.lastNeed), this.lastChar.toString(this.encoding, 0, this.lastTotal)) : (e3.copy(this.lastChar, t3, 0, e3.length), void (this.lastNeed -= e3.length));
        }
        function c(e3, t3) {
          if ((e3.length - t3) % 2 == 0) {
            var r3 = e3.toString("utf16le", t3);
            if (r3) {
              var n3 = r3.charCodeAt(r3.length - 1);
              if (n3 >= 55296 && n3 <= 56319)
                return this.lastNeed = 2, this.lastTotal = 4, this.lastChar[0] = e3[e3.length - 2], this.lastChar[1] = e3[e3.length - 1], r3.slice(0, -1);
            }
            return r3;
          }
          return this.lastNeed = 1, this.lastTotal = 2, this.lastChar[0] = e3[e3.length - 1], e3.toString("utf16le", t3, e3.length - 1);
        }
        function u(e3) {
          var t3 = e3 && e3.length ? this.write(e3) : "";
          if (this.lastNeed) {
            var r3 = this.lastTotal - this.lastNeed;
            return t3 + this.lastChar.toString("utf16le", 0, r3);
          }
          return t3;
        }
        function l(e3, t3) {
          var r3 = (e3.length - t3) % 3;
          return 0 === r3 ? e3.toString("base64", t3) : (this.lastNeed = 3 - r3, this.lastTotal = 3, 1 === r3 ? this.lastChar[0] = e3[e3.length - 1] : (this.lastChar[0] = e3[e3.length - 2], this.lastChar[1] = e3[e3.length - 1]), e3.toString("base64", t3, e3.length - r3));
        }
        function h(e3) {
          var t3 = e3 && e3.length ? this.write(e3) : "";
          return this.lastNeed ? t3 + this.lastChar.toString("base64", 0, 3 - this.lastNeed) : t3;
        }
        function p(e3) {
          return e3.toString(this.encoding);
        }
        function d(e3) {
          return e3 && e3.length ? this.write(e3) : "";
        }
        t2.s = i, i.prototype.write = function(e3) {
          if (0 === e3.length)
            return "";
          var t3, r3;
          if (this.lastNeed) {
            if (void 0 === (t3 = this.fillLast(e3)))
              return "";
            r3 = this.lastNeed, this.lastNeed = 0;
          } else
            r3 = 0;
          return r3 < e3.length ? t3 ? t3 + this.text(e3, r3) : this.text(e3, r3) : t3 || "";
        }, i.prototype.end = function(e3) {
          var t3 = e3 && e3.length ? this.write(e3) : "";
          return this.lastNeed ? t3 + "\uFFFD" : t3;
        }, i.prototype.text = function(e3, t3) {
          var r3 = function(e4, t4, r4) {
            var n4 = t4.length - 1;
            if (n4 < r4)
              return 0;
            var s2 = o(t4[n4]);
            return s2 >= 0 ? (s2 > 0 && (e4.lastNeed = s2 - 1), s2) : --n4 < r4 || -2 === s2 ? 0 : (s2 = o(t4[n4])) >= 0 ? (s2 > 0 && (e4.lastNeed = s2 - 2), s2) : --n4 < r4 || -2 === s2 ? 0 : (s2 = o(t4[n4])) >= 0 ? (s2 > 0 && (2 === s2 ? s2 = 0 : e4.lastNeed = s2 - 3), s2) : 0;
          }(this, e3, t3);
          if (!this.lastNeed)
            return e3.toString("utf8", t3);
          this.lastTotal = r3;
          var n3 = e3.length - (r3 - this.lastNeed);
          return e3.copy(this.lastChar, 0, n3), e3.toString("utf8", t3, n3);
        }, i.prototype.fillLast = function(e3) {
          if (this.lastNeed <= e3.length)
            return e3.copy(this.lastChar, this.lastTotal - this.lastNeed, 0, this.lastNeed), this.lastChar.toString(this.encoding, 0, this.lastTotal);
          e3.copy(this.lastChar, this.lastTotal - this.lastNeed, 0, e3.length), this.lastNeed -= e3.length;
        };
      }, 5457: (e2, t2, r2) => {
        "use strict";
        r2.d(t2, { vw: () => i, rq: () => s, EL: () => a, NY: () => o });
        let n2 = 0;
        const s = (e3) => Math.floor(e3 / 25.4 * 72 * 20), i = (e3) => Math.floor(72 * e3 * 20), o = () => ++n2, a = () => ((e3 = 21) => {
          let t3 = "", r3 = e3;
          for (; r3--; )
            t3 += "useandom-26T198340PX75pxJACKVERYMINDBUSHWOLF_GQZbfghjklqvwyzrict"[64 * Math.random() | 0];
          return t3;
        })().toLowerCase();
      }, 5575: (e2, t2, r2) => {
        "use strict";
        r2.r(t2), r2.d(t2, { convertInchesToTwip: () => n2.vw, convertMillimetersToTwip: () => n2.rq, dateTimeValue: () => s.sF, decimalNumber: () => s.vH, eighthPointMeasureValue: () => s.LV, hexColorValue: () => s.dg, hpsMeasureValue: () => s.KR, longHexNumber: () => s.mA, measurementOrPercentValue: () => s.aB, percentageValue: () => s.wp, pointMeasureValue: () => s.gg, positiveUniversalMeasureValue: () => s._p, shortHexNumber: () => s.G0, signedHpsMeasureValue: () => s.Rg, signedTwipsMeasureValue: () => s.xb, twipsMeasureValue: () => s.Jd, uCharHexNumber: () => s.xD, uniqueId: () => n2.EL, uniqueNumericId: () => n2.NY, universalMeasureValue: () => s.KC, unsignedDecimalNumber: () => s.f$ });
        var n2 = r2(5457), s = r2(6595);
      }, 6595: (e2, t2, r2) => {
        "use strict";
        r2.d(t2, { G0: () => a, Jd: () => w, KC: () => u, KR: () => f, LV: () => b, Rg: () => m, _p: () => h, aB: () => y, dg: () => p, f$: () => s, gg: () => x, mA: () => o, sF: () => v, vH: () => n2, wp: () => g, xD: () => c, xb: () => d });
        const n2 = (e3) => {
          if (isNaN(e3))
            throw new Error(`Invalid value '${e3}' specified. Must be an integer.`);
          return Math.floor(e3);
        }, s = (e3) => {
          const t3 = n2(e3);
          if (t3 < 0)
            throw new Error(`Invalid value '${e3}' specified. Must be a positive integer.`);
          return t3;
        }, i = (e3, t3) => {
          const r3 = 2 * t3;
          if (e3.length !== r3 || isNaN(Number(`0x${e3}`)))
            throw new Error(`Invalid hex value '${e3}'. Expected ${r3} digit hex value`);
          return e3;
        }, o = (e3) => i(e3, 4), a = (e3) => i(e3, 2), c = (e3) => i(e3, 1), u = (e3) => {
          const t3 = e3.slice(-2);
          if (!l.includes(t3))
            throw new Error(`Invalid unit '${t3}' specified. Valid units are ${l.join(", ")}`);
          const r3 = e3.substring(0, e3.length - 2);
          if (isNaN(Number(r3)))
            throw new Error(`Invalid value '${r3}' specified. Expected a valid number.`);
          return `${Number(r3)}${t3}`;
        }, l = ["mm", "cm", "in", "pt", "pc", "pi"], h = (e3) => {
          const t3 = u(e3);
          if (parseFloat(t3) < 0)
            throw new Error(`Invalid value '${t3}' specified. Expected a positive number.`);
          return t3;
        }, p = (e3) => {
          if ("auto" === e3)
            return e3;
          const t3 = "#" === e3.charAt(0) ? e3.substring(1) : e3;
          return i(t3, 3);
        }, d = (e3) => "string" == typeof e3 ? u(e3) : n2(e3), f = (e3) => "string" == typeof e3 ? h(e3) : s(e3), m = (e3) => "string" == typeof e3 ? u(e3) : n2(e3), w = (e3) => "string" == typeof e3 ? h(e3) : s(e3), g = (e3) => {
          if ("%" !== e3.slice(-1))
            throw new Error(`Invalid value '${e3}'. Expected percentage value (eg '55%')`);
          const t3 = e3.substring(0, e3.length - 1);
          if (isNaN(Number(t3)))
            throw new Error(`Invalid value '${t3}' specified. Expected a valid number.`);
          return `${Number(t3)}%`;
        }, y = (e3) => "number" == typeof e3 ? n2(e3) : "%" === e3.slice(-1) ? g(e3) : u(e3), b = s, x = s, v = (e3) => e3.toISOString();
      }, 4927: (e2, t2, r2) => {
        function n2(e3) {
          try {
            if (!r2.g.localStorage)
              return false;
          } catch (e4) {
            return false;
          }
          var t3 = r2.g.localStorage[e3];
          return null != t3 && "true" === String(t3).toLowerCase();
        }
        e2.exports = function(e3, t3) {
          if (n2("noDeprecation"))
            return e3;
          var r3 = false;
          return function() {
            if (!r3) {
              if (n2("throwDeprecation"))
                throw new Error(t3);
              n2("traceDeprecation") ? console.trace(t3) : console.warn(t3), r3 = true;
            }
            return e3.apply(this, arguments);
          };
        };
      }, 9881: (e2) => {
        e2.exports = { isArray: function(e3) {
          return Array.isArray ? Array.isArray(e3) : "[object Array]" === Object.prototype.toString.call(e3);
        } };
      }, 7888: (e2, t2, r2) => {
        var n2 = r2(1229), s = r2(1388), i = r2(6501), o = r2(4673);
        e2.exports = { xml2js: n2, xml2json: s, js2xml: i, json2xml: o };
      }, 6501: (e2, t2, r2) => {
        var n2, s, i = r2(4740), o = r2(9881).isArray;
        function a(e3, t3, r3) {
          return (!r3 && e3.spaces ? "\n" : "") + Array(t3 + 1).join(e3.spaces);
        }
        function c(e3, t3, r3) {
          if (t3.ignoreAttributes)
            return "";
          "attributesFn" in t3 && (e3 = t3.attributesFn(e3, s, n2));
          var i2, o2, c2, u2, l2 = [];
          for (i2 in e3)
            e3.hasOwnProperty(i2) && null !== e3[i2] && void 0 !== e3[i2] && (u2 = t3.noQuotesForNativeAttributes && "string" != typeof e3[i2] ? "" : '"', o2 = (o2 = "" + e3[i2]).replace(/"/g, "&quot;"), c2 = "attributeNameFn" in t3 ? t3.attributeNameFn(i2, o2, s, n2) : i2, l2.push(t3.spaces && t3.indentAttributes ? a(t3, r3 + 1, false) : " "), l2.push(c2 + "=" + u2 + ("attributeValueFn" in t3 ? t3.attributeValueFn(o2, i2, s, n2) : o2) + u2));
          return e3 && Object.keys(e3).length && t3.spaces && t3.indentAttributes && l2.push(a(t3, r3, false)), l2.join("");
        }
        function u(e3, t3, r3) {
          return n2 = e3, s = "xml", t3.ignoreDeclaration ? "" : "<?xml" + c(e3[t3.attributesKey], t3, r3) + "?>";
        }
        function l(e3, t3, r3) {
          if (t3.ignoreInstruction)
            return "";
          var i2;
          for (i2 in e3)
            if (e3.hasOwnProperty(i2))
              break;
          var o2 = "instructionNameFn" in t3 ? t3.instructionNameFn(i2, e3[i2], s, n2) : i2;
          if ("object" == typeof e3[i2])
            return n2 = e3, s = o2, "<?" + o2 + c(e3[i2][t3.attributesKey], t3, r3) + "?>";
          var a2 = e3[i2] ? e3[i2] : "";
          return "instructionFn" in t3 && (a2 = t3.instructionFn(a2, i2, s, n2)), "<?" + o2 + (a2 ? " " + a2 : "") + "?>";
        }
        function h(e3, t3) {
          return t3.ignoreComment ? "" : "<!--" + ("commentFn" in t3 ? t3.commentFn(e3, s, n2) : e3) + "-->";
        }
        function p(e3, t3) {
          return t3.ignoreCdata ? "" : "<![CDATA[" + ("cdataFn" in t3 ? t3.cdataFn(e3, s, n2) : e3.replace("]]>", "]]]]><![CDATA[>")) + "]]>";
        }
        function d(e3, t3) {
          return t3.ignoreDoctype ? "" : "<!DOCTYPE " + ("doctypeFn" in t3 ? t3.doctypeFn(e3, s, n2) : e3) + ">";
        }
        function f(e3, t3) {
          return t3.ignoreText ? "" : (e3 = (e3 = (e3 = "" + e3).replace(/&amp;/g, "&")).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"), "textFn" in t3 ? t3.textFn(e3, s, n2) : e3);
        }
        function m(e3, t3, r3, i2) {
          return e3.reduce(function(e4, o2) {
            var u2 = a(t3, r3, i2 && !e4);
            switch (o2.type) {
              case "element":
                return e4 + u2 + function(e5, t4, r4) {
                  n2 = e5, s = e5.name;
                  var i3 = [], o3 = "elementNameFn" in t4 ? t4.elementNameFn(e5.name, e5) : e5.name;
                  i3.push("<" + o3), e5[t4.attributesKey] && i3.push(c(e5[t4.attributesKey], t4, r4));
                  var a2 = e5[t4.elementsKey] && e5[t4.elementsKey].length || e5[t4.attributesKey] && "preserve" === e5[t4.attributesKey]["xml:space"];
                  return a2 || (a2 = "fullTagEmptyElementFn" in t4 ? t4.fullTagEmptyElementFn(e5.name, e5) : t4.fullTagEmptyElement), a2 ? (i3.push(">"), e5[t4.elementsKey] && e5[t4.elementsKey].length && (i3.push(m(e5[t4.elementsKey], t4, r4 + 1)), n2 = e5, s = e5.name), i3.push(t4.spaces && function(e6, t5) {
                    var r5;
                    if (e6.elements && e6.elements.length)
                      for (r5 = 0; r5 < e6.elements.length; ++r5)
                        switch (e6.elements[r5][t5.typeKey]) {
                          case "text":
                            if (t5.indentText)
                              return true;
                            break;
                          case "cdata":
                            if (t5.indentCdata)
                              return true;
                            break;
                          case "instruction":
                            if (t5.indentInstruction)
                              return true;
                            break;
                          default:
                            return true;
                        }
                    return false;
                  }(e5, t4) ? "\n" + Array(r4 + 1).join(t4.spaces) : ""), i3.push("</" + o3 + ">")) : i3.push("/>"), i3.join("");
                }(o2, t3, r3);
              case "comment":
                return e4 + u2 + h(o2[t3.commentKey], t3);
              case "doctype":
                return e4 + u2 + d(o2[t3.doctypeKey], t3);
              case "cdata":
                return e4 + (t3.indentCdata ? u2 : "") + p(o2[t3.cdataKey], t3);
              case "text":
                return e4 + (t3.indentText ? u2 : "") + f(o2[t3.textKey], t3);
              case "instruction":
                var w2 = {};
                return w2[o2[t3.nameKey]] = o2[t3.attributesKey] ? o2 : o2[t3.instructionKey], e4 + (t3.indentInstruction ? u2 : "") + l(w2, t3, r3);
            }
          }, "");
        }
        function w(e3, t3, r3) {
          var n3;
          for (n3 in e3)
            if (e3.hasOwnProperty(n3))
              switch (n3) {
                case t3.parentKey:
                case t3.attributesKey:
                  break;
                case t3.textKey:
                  if (t3.indentText || r3)
                    return true;
                  break;
                case t3.cdataKey:
                  if (t3.indentCdata || r3)
                    return true;
                  break;
                case t3.instructionKey:
                  if (t3.indentInstruction || r3)
                    return true;
                  break;
                case t3.doctypeKey:
                case t3.commentKey:
                default:
                  return true;
              }
          return false;
        }
        function g(e3, t3, r3, i2, o2) {
          n2 = e3, s = t3;
          var u2 = "elementNameFn" in r3 ? r3.elementNameFn(t3, e3) : t3;
          if (null == e3 || "" === e3)
            return "fullTagEmptyElementFn" in r3 && r3.fullTagEmptyElementFn(t3, e3) || r3.fullTagEmptyElement ? "<" + u2 + "></" + u2 + ">" : "<" + u2 + "/>";
          var l2 = [];
          if (t3) {
            if (l2.push("<" + u2), "object" != typeof e3)
              return l2.push(">" + f(e3, r3) + "</" + u2 + ">"), l2.join("");
            e3[r3.attributesKey] && l2.push(c(e3[r3.attributesKey], r3, i2));
            var h2 = w(e3, r3, true) || e3[r3.attributesKey] && "preserve" === e3[r3.attributesKey]["xml:space"];
            if (h2 || (h2 = "fullTagEmptyElementFn" in r3 ? r3.fullTagEmptyElementFn(t3, e3) : r3.fullTagEmptyElement), !h2)
              return l2.push("/>"), l2.join("");
            l2.push(">");
          }
          return l2.push(y(e3, r3, i2 + 1, false)), n2 = e3, s = t3, t3 && l2.push((o2 ? a(r3, i2, false) : "") + "</" + u2 + ">"), l2.join("");
        }
        function y(e3, t3, r3, n3) {
          var s2, i2, c2, m2 = [];
          for (i2 in e3)
            if (e3.hasOwnProperty(i2))
              for (c2 = o(e3[i2]) ? e3[i2] : [e3[i2]], s2 = 0; s2 < c2.length; ++s2) {
                switch (i2) {
                  case t3.declarationKey:
                    m2.push(u(c2[s2], t3, r3));
                    break;
                  case t3.instructionKey:
                    m2.push((t3.indentInstruction ? a(t3, r3, n3) : "") + l(c2[s2], t3, r3));
                    break;
                  case t3.attributesKey:
                  case t3.parentKey:
                    break;
                  case t3.textKey:
                    m2.push((t3.indentText ? a(t3, r3, n3) : "") + f(c2[s2], t3));
                    break;
                  case t3.cdataKey:
                    m2.push((t3.indentCdata ? a(t3, r3, n3) : "") + p(c2[s2], t3));
                    break;
                  case t3.doctypeKey:
                    m2.push(a(t3, r3, n3) + d(c2[s2], t3));
                    break;
                  case t3.commentKey:
                    m2.push(a(t3, r3, n3) + h(c2[s2], t3));
                    break;
                  default:
                    m2.push(a(t3, r3, n3) + g(c2[s2], i2, t3, r3, w(c2[s2], t3)));
                }
                n3 = n3 && !m2.length;
              }
          return m2.join("");
        }
        e2.exports = function(e3, t3) {
          t3 = function(e4) {
            var t4 = i.copyOptions(e4);
            return i.ensureFlagExists("ignoreDeclaration", t4), i.ensureFlagExists("ignoreInstruction", t4), i.ensureFlagExists("ignoreAttributes", t4), i.ensureFlagExists("ignoreText", t4), i.ensureFlagExists("ignoreComment", t4), i.ensureFlagExists("ignoreCdata", t4), i.ensureFlagExists("ignoreDoctype", t4), i.ensureFlagExists("compact", t4), i.ensureFlagExists("indentText", t4), i.ensureFlagExists("indentCdata", t4), i.ensureFlagExists("indentAttributes", t4), i.ensureFlagExists("indentInstruction", t4), i.ensureFlagExists("fullTagEmptyElement", t4), i.ensureFlagExists("noQuotesForNativeAttributes", t4), i.ensureSpacesExists(t4), "number" == typeof t4.spaces && (t4.spaces = Array(t4.spaces + 1).join(" ")), i.ensureKeyExists("declaration", t4), i.ensureKeyExists("instruction", t4), i.ensureKeyExists("attributes", t4), i.ensureKeyExists("text", t4), i.ensureKeyExists("comment", t4), i.ensureKeyExists("cdata", t4), i.ensureKeyExists("doctype", t4), i.ensureKeyExists("type", t4), i.ensureKeyExists("name", t4), i.ensureKeyExists("elements", t4), i.checkFnExists("doctype", t4), i.checkFnExists("instruction", t4), i.checkFnExists("cdata", t4), i.checkFnExists("comment", t4), i.checkFnExists("text", t4), i.checkFnExists("instructionName", t4), i.checkFnExists("elementName", t4), i.checkFnExists("attributeName", t4), i.checkFnExists("attributeValue", t4), i.checkFnExists("attributes", t4), i.checkFnExists("fullTagEmptyElement", t4), t4;
          }(t3);
          var r3 = [];
          return n2 = e3, s = "_root_", t3.compact ? r3.push(y(e3, t3, 0, true)) : (e3[t3.declarationKey] && r3.push(u(e3[t3.declarationKey], t3, 0)), e3[t3.elementsKey] && e3[t3.elementsKey].length && r3.push(m(e3[t3.elementsKey], t3, 0, !r3.length))), r3.join("");
        };
      }, 4673: (e2, t2, r2) => {
        var n2 = r2(6501);
        e2.exports = function(e3, t3) {
          e3 instanceof Buffer && (e3 = e3.toString());
          var r3 = null;
          if ("string" == typeof e3)
            try {
              r3 = JSON.parse(e3);
            } catch (e4) {
              throw new Error("The JSON structure is invalid");
            }
          else
            r3 = e3;
          return n2(r3, t3);
        };
      }, 4740: (e2, t2, r2) => {
        var n2 = r2(9881).isArray;
        e2.exports = { copyOptions: function(e3) {
          var t3, r3 = {};
          for (t3 in e3)
            e3.hasOwnProperty(t3) && (r3[t3] = e3[t3]);
          return r3;
        }, ensureFlagExists: function(e3, t3) {
          e3 in t3 && "boolean" == typeof t3[e3] || (t3[e3] = false);
        }, ensureSpacesExists: function(e3) {
          (!("spaces" in e3) || "number" != typeof e3.spaces && "string" != typeof e3.spaces) && (e3.spaces = 0);
        }, ensureAlwaysArrayExists: function(e3) {
          "alwaysArray" in e3 && ("boolean" == typeof e3.alwaysArray || n2(e3.alwaysArray)) || (e3.alwaysArray = false);
        }, ensureKeyExists: function(e3, t3) {
          e3 + "Key" in t3 && "string" == typeof t3[e3 + "Key"] || (t3[e3 + "Key"] = t3.compact ? "_" + e3 : e3);
        }, checkFnExists: function(e3, t3) {
          return e3 + "Fn" in t3;
        } };
      }, 1229: (e2, t2, r2) => {
        var n2, s, i = r2(6099), o = r2(4740), a = r2(9881).isArray;
        function c(e3) {
          var t3 = Number(e3);
          if (!isNaN(t3))
            return t3;
          var r3 = e3.toLowerCase();
          return "true" === r3 || "false" !== r3 && e3;
        }
        function u(e3, t3) {
          var r3;
          if (n2.compact) {
            if (!s[n2[e3 + "Key"]] && (a(n2.alwaysArray) ? -1 !== n2.alwaysArray.indexOf(n2[e3 + "Key"]) : n2.alwaysArray) && (s[n2[e3 + "Key"]] = []), s[n2[e3 + "Key"]] && !a(s[n2[e3 + "Key"]]) && (s[n2[e3 + "Key"]] = [s[n2[e3 + "Key"]]]), e3 + "Fn" in n2 && "string" == typeof t3 && (t3 = n2[e3 + "Fn"](t3, s)), "instruction" === e3 && ("instructionFn" in n2 || "instructionNameFn" in n2)) {
              for (r3 in t3)
                if (t3.hasOwnProperty(r3))
                  if ("instructionFn" in n2)
                    t3[r3] = n2.instructionFn(t3[r3], r3, s);
                  else {
                    var i2 = t3[r3];
                    delete t3[r3], t3[n2.instructionNameFn(r3, i2, s)] = i2;
                  }
            }
            a(s[n2[e3 + "Key"]]) ? s[n2[e3 + "Key"]].push(t3) : s[n2[e3 + "Key"]] = t3;
          } else {
            s[n2.elementsKey] || (s[n2.elementsKey] = []);
            var o2 = {};
            if (o2[n2.typeKey] = e3, "instruction" === e3) {
              for (r3 in t3)
                if (t3.hasOwnProperty(r3))
                  break;
              o2[n2.nameKey] = "instructionNameFn" in n2 ? n2.instructionNameFn(r3, t3, s) : r3, n2.instructionHasAttributes ? (o2[n2.attributesKey] = t3[r3][n2.attributesKey], "instructionFn" in n2 && (o2[n2.attributesKey] = n2.instructionFn(o2[n2.attributesKey], r3, s))) : ("instructionFn" in n2 && (t3[r3] = n2.instructionFn(t3[r3], r3, s)), o2[n2.instructionKey] = t3[r3]);
            } else
              e3 + "Fn" in n2 && (t3 = n2[e3 + "Fn"](t3, s)), o2[n2[e3 + "Key"]] = t3;
            n2.addParent && (o2[n2.parentKey] = s), s[n2.elementsKey].push(o2);
          }
        }
        function l(e3) {
          var t3;
          if ("attributesFn" in n2 && e3 && (e3 = n2.attributesFn(e3, s)), (n2.trim || "attributeValueFn" in n2 || "attributeNameFn" in n2 || n2.nativeTypeAttributes) && e3) {
            for (t3 in e3)
              if (e3.hasOwnProperty(t3) && (n2.trim && (e3[t3] = e3[t3].trim()), n2.nativeTypeAttributes && (e3[t3] = c(e3[t3])), "attributeValueFn" in n2 && (e3[t3] = n2.attributeValueFn(e3[t3], t3, s)), "attributeNameFn" in n2)) {
                var r3 = e3[t3];
                delete e3[t3], e3[n2.attributeNameFn(t3, e3[t3], s)] = r3;
              }
          }
          return e3;
        }
        function h(e3) {
          var t3 = {};
          if (e3.body && ("xml" === e3.name.toLowerCase() || n2.instructionHasAttributes)) {
            for (var r3, i2 = /([\w:-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|(\w+))\s*/g; null !== (r3 = i2.exec(e3.body)); )
              t3[r3[1]] = r3[2] || r3[3] || r3[4];
            t3 = l(t3);
          }
          if ("xml" === e3.name.toLowerCase()) {
            if (n2.ignoreDeclaration)
              return;
            s[n2.declarationKey] = {}, Object.keys(t3).length && (s[n2.declarationKey][n2.attributesKey] = t3), n2.addParent && (s[n2.declarationKey][n2.parentKey] = s);
          } else {
            if (n2.ignoreInstruction)
              return;
            n2.trim && (e3.body = e3.body.trim());
            var o2 = {};
            n2.instructionHasAttributes && Object.keys(t3).length ? (o2[e3.name] = {}, o2[e3.name][n2.attributesKey] = t3) : o2[e3.name] = e3.body, u("instruction", o2);
          }
        }
        function p(e3, t3) {
          var r3;
          if ("object" == typeof e3 && (t3 = e3.attributes, e3 = e3.name), t3 = l(t3), "elementNameFn" in n2 && (e3 = n2.elementNameFn(e3, s)), n2.compact) {
            var i2;
            if (r3 = {}, !n2.ignoreAttributes && t3 && Object.keys(t3).length)
              for (i2 in r3[n2.attributesKey] = {}, t3)
                t3.hasOwnProperty(i2) && (r3[n2.attributesKey][i2] = t3[i2]);
            !(e3 in s) && (a(n2.alwaysArray) ? -1 !== n2.alwaysArray.indexOf(e3) : n2.alwaysArray) && (s[e3] = []), s[e3] && !a(s[e3]) && (s[e3] = [s[e3]]), a(s[e3]) ? s[e3].push(r3) : s[e3] = r3;
          } else
            s[n2.elementsKey] || (s[n2.elementsKey] = []), (r3 = {})[n2.typeKey] = "element", r3[n2.nameKey] = e3, !n2.ignoreAttributes && t3 && Object.keys(t3).length && (r3[n2.attributesKey] = t3), n2.alwaysChildren && (r3[n2.elementsKey] = []), s[n2.elementsKey].push(r3);
          r3[n2.parentKey] = s, s = r3;
        }
        function d(e3) {
          n2.ignoreText || (e3.trim() || n2.captureSpacesBetweenElements) && (n2.trim && (e3 = e3.trim()), n2.nativeType && (e3 = c(e3)), n2.sanitize && (e3 = e3.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")), u("text", e3));
        }
        function f(e3) {
          n2.ignoreComment || (n2.trim && (e3 = e3.trim()), u("comment", e3));
        }
        function m(e3) {
          var t3 = s[n2.parentKey];
          n2.addParent || delete s[n2.parentKey], s = t3;
        }
        function w(e3) {
          n2.ignoreCdata || (n2.trim && (e3 = e3.trim()), u("cdata", e3));
        }
        function g(e3) {
          n2.ignoreDoctype || (e3 = e3.replace(/^ /, ""), n2.trim && (e3 = e3.trim()), u("doctype", e3));
        }
        function y(e3) {
          e3.note = e3;
        }
        e2.exports = function(e3, t3) {
          var r3 = i.parser(true, {}), a2 = {};
          if (s = a2, n2 = function(e4) {
            return n2 = o.copyOptions(e4), o.ensureFlagExists("ignoreDeclaration", n2), o.ensureFlagExists("ignoreInstruction", n2), o.ensureFlagExists("ignoreAttributes", n2), o.ensureFlagExists("ignoreText", n2), o.ensureFlagExists("ignoreComment", n2), o.ensureFlagExists("ignoreCdata", n2), o.ensureFlagExists("ignoreDoctype", n2), o.ensureFlagExists("compact", n2), o.ensureFlagExists("alwaysChildren", n2), o.ensureFlagExists("addParent", n2), o.ensureFlagExists("trim", n2), o.ensureFlagExists("nativeType", n2), o.ensureFlagExists("nativeTypeAttributes", n2), o.ensureFlagExists("sanitize", n2), o.ensureFlagExists("instructionHasAttributes", n2), o.ensureFlagExists("captureSpacesBetweenElements", n2), o.ensureAlwaysArrayExists(n2), o.ensureKeyExists("declaration", n2), o.ensureKeyExists("instruction", n2), o.ensureKeyExists("attributes", n2), o.ensureKeyExists("text", n2), o.ensureKeyExists("comment", n2), o.ensureKeyExists("cdata", n2), o.ensureKeyExists("doctype", n2), o.ensureKeyExists("type", n2), o.ensureKeyExists("name", n2), o.ensureKeyExists("elements", n2), o.ensureKeyExists("parent", n2), o.checkFnExists("doctype", n2), o.checkFnExists("instruction", n2), o.checkFnExists("cdata", n2), o.checkFnExists("comment", n2), o.checkFnExists("text", n2), o.checkFnExists("instructionName", n2), o.checkFnExists("elementName", n2), o.checkFnExists("attributeName", n2), o.checkFnExists("attributeValue", n2), o.checkFnExists("attributes", n2), n2;
          }(t3), r3.opt = { strictEntities: true }, r3.onopentag = p, r3.ontext = d, r3.oncomment = f, r3.onclosetag = m, r3.onerror = y, r3.oncdata = w, r3.ondoctype = g, r3.onprocessinginstruction = h, r3.write(e3).close(), a2[n2.elementsKey]) {
            var c2 = a2[n2.elementsKey];
            delete a2[n2.elementsKey], a2[n2.elementsKey] = c2, delete a2.text;
          }
          return a2;
        };
      }, 1388: (e2, t2, r2) => {
        var n2 = r2(4740), s = r2(1229);
        e2.exports = function(e3, t3) {
          var r3, i, o;
          return r3 = function(e4) {
            var t4 = n2.copyOptions(e4);
            return n2.ensureSpacesExists(t4), t4;
          }(t3), i = s(e3, r3), o = "compact" in r3 && r3.compact ? "_parent" : "parent", ("addParent" in r3 && r3.addParent ? JSON.stringify(i, function(e4, t4) {
            return e4 === o ? "_" : t4;
          }, r3.spaces) : JSON.stringify(i, null, r3.spaces)).replace(/\u2028/g, "\\u2028").replace(/\u2029/g, "\\u2029");
        };
      }, 255: (e2) => {
        var t2 = { "&": "&amp;", '"': "&quot;", "'": "&apos;", "<": "&lt;", ">": "&gt;" };
        e2.exports = function(e3) {
          return e3 && e3.replace ? e3.replace(/([&"<>'])/g, function(e4, r2) {
            return t2[r2];
          }) : e3;
        };
      }, 3479: (e2, t2, r2) => {
        var n2 = r2(4155), s = r2(255), i = r2(2830).Stream;
        function o(e3, t3, r3) {
          r3 = r3 || 0;
          var n3, i2, a2 = (n3 = t3, new Array(r3 || 0).join(n3 || "")), c = e3;
          if ("object" == typeof e3 && (c = e3[i2 = Object.keys(e3)[0]]) && c._elem)
            return c._elem.name = i2, c._elem.icount = r3, c._elem.indent = t3, c._elem.indents = a2, c._elem.interrupt = c, c._elem;
          var u, l = [], h = [];
          function p(e4) {
            Object.keys(e4).forEach(function(t4) {
              l.push(function(e5, t5) {
                return e5 + '="' + s(t5) + '"';
              }(t4, e4[t4]));
            });
          }
          switch (typeof c) {
            case "object":
              if (null === c)
                break;
              c._attr && p(c._attr), c._cdata && h.push(("<![CDATA[" + c._cdata).replace(/\]\]>/g, "]]]]><![CDATA[>") + "]]>"), c.forEach && (u = false, h.push(""), c.forEach(function(e4) {
                "object" == typeof e4 ? "_attr" == Object.keys(e4)[0] ? p(e4._attr) : h.push(o(e4, t3, r3 + 1)) : (h.pop(), u = true, h.push(s(e4)));
              }), u || h.push(""));
              break;
            default:
              h.push(s(c));
          }
          return { name: i2, interrupt: false, attributes: l, content: h, icount: r3, indents: a2, indent: t3 };
        }
        function a(e3, t3, r3) {
          if ("object" != typeof t3)
            return e3(false, t3);
          var n3 = t3.interrupt ? 1 : t3.content.length;
          function s2() {
            for (; t3.content.length; ) {
              var s3 = t3.content.shift();
              if (void 0 !== s3) {
                if (i2(s3))
                  return;
                a(e3, s3);
              }
            }
            e3(false, (n3 > 1 ? t3.indents : "") + (t3.name ? "</" + t3.name + ">" : "") + (t3.indent && !r3 ? "\n" : "")), r3 && r3();
          }
          function i2(t4) {
            return !!t4.interrupt && (t4.interrupt.append = e3, t4.interrupt.end = s2, t4.interrupt = false, e3(true), true);
          }
          if (e3(false, t3.indents + (t3.name ? "<" + t3.name : "") + (t3.attributes.length ? " " + t3.attributes.join(" ") : "") + (n3 ? t3.name ? ">" : "" : t3.name ? "/>" : "") + (t3.indent && n3 > 1 ? "\n" : "")), !n3)
            return e3(false, t3.indent ? "\n" : "");
          i2(t3) || s2();
        }
        e2.exports = function(e3, t3) {
          "object" != typeof t3 && (t3 = { indent: t3 });
          var r3, s2, c = t3.stream ? new i() : null, u = "", l = false, h = t3.indent ? true === t3.indent ? "    " : t3.indent : "", p = true;
          function d(e4) {
            p ? n2.nextTick(e4) : e4();
          }
          function f(e4, t4) {
            if (void 0 !== t4 && (u += t4), e4 && !l && (c = c || new i(), l = true), e4 && l) {
              var r4 = u;
              d(function() {
                c.emit("data", r4);
              }), u = "";
            }
          }
          function m(e4, t4) {
            a(f, o(e4, h, h ? 1 : 0), t4);
          }
          function w() {
            if (c) {
              var e4 = u;
              d(function() {
                c.emit("data", e4), c.emit("end"), c.readable = false, c.emit("close");
              });
            }
          }
          return d(function() {
            p = false;
          }), t3.declaration && (s2 = { version: "1.0", encoding: (r3 = t3.declaration).encoding || "UTF-8" }, r3.standalone && (s2.standalone = r3.standalone), m({ "?xml": { _attr: s2 } }), u = u.replace("/>", "?>")), e3 && e3.forEach ? e3.forEach(function(t4, r4) {
            var n3;
            r4 + 1 === e3.length && (n3 = w), m(t4, n3);
          }) : m(e3, w), c ? (c.readable = true, c) : u;
        }, e2.exports.element = e2.exports.Element = function() {
          var e3 = Array.prototype.slice.call(arguments), t3 = { _elem: o(e3), push: function(e4) {
            if (!this.append)
              throw new Error("not assigned to a parent!");
            var t4 = this, r3 = this._elem.indent;
            a(this.append, o(e4, r3, this._elem.icount + (r3 ? 1 : 0)), function() {
              t4.append(true);
            });
          }, close: function(e4) {
            void 0 !== e4 && this.push(e4), this.end && this.end();
          } };
          return t3;
        };
      } }, t = {};
      function r(n2) {
        var s = t[n2];
        if (void 0 !== s)
          return s.exports;
        var i = t[n2] = { exports: {} };
        return e[n2].call(i.exports, i, i.exports, r), i.exports;
      }
      r.d = (e2, t2) => {
        for (var n2 in t2)
          r.o(t2, n2) && !r.o(e2, n2) && Object.defineProperty(e2, n2, { enumerable: true, get: t2[n2] });
      }, r.g = function() {
        if ("object" == typeof globalThis)
          return globalThis;
        try {
          return this || new Function("return this")();
        } catch (e2) {
          if ("object" == typeof window)
            return window;
        }
      }(), r.o = (e2, t2) => Object.prototype.hasOwnProperty.call(e2, t2), r.r = (e2) => {
        "undefined" != typeof Symbol && Symbol.toStringTag && Object.defineProperty(e2, Symbol.toStringTag, { value: "Module" }), Object.defineProperty(e2, "__esModule", { value: true });
      };
      var n = {};
      return (() => {
        "use strict";
        r.r(n), r.d(n, { AbstractNumbering: () => Ji, Alignment: () => D, AlignmentAttributes: () => L, AlignmentType: () => m, Attributes: () => a, BaseEmphasisMark: () => Q, BaseXmlComponent: () => e2, Body: () => un, Bookmark: () => Or, BookmarkEnd: () => Lr, BookmarkStart: () => kr, Border: () => B, BorderElement: () => P, BorderStyle: () => w, Column: () => hn, ColumnAttributes: () => ln, ColumnBreak: () => pr, Columns: () => jr, ColumnsAttributes: () => zr, Comment: () => cr, CommentRangeEnd: () => or, CommentRangeStart: () => ir, CommentReference: () => ar, Comments: () => ur, ConcreteHyperlink: () => Sr, ConcreteNumbering: () => to, DeletedTextRun: () => na, DocGridAttributes: () => Wr, Document: () => Ho, DocumentAttributes: () => pn, DocumentBackground: () => fn, DocumentBackgroundAttributes: () => dn, DocumentDefaults: () => Bo, DocumentGrid: () => Kr, DocumentGridType: () => Pe, DotEmphasisMark: () => te, Drawing: () => Xt, DropCapType: () => vn, EMPTY_OBJECT: () => t2, EmphasisMark: () => ee, EmphasisMarkType: () => E, ExternalHyperlink: () => Rr, File: () => Ho, FootNoteReferenceRunAttributes: () => qo, FootNotes: () => ki, Footer: () => Xo, FooterWrapper: () => _i, FootnoteReference: () => Zo, FootnoteReferenceRun: () => Yo, FrameAnchorType: () => _n, FrameProperties: () => An, FramePropertiesAttributes: () => Tn, FrameWrap: () => En, GridSpan: () => ks, Header: () => $o, HeaderFooterReference: () => Hr, HeaderFooterReferenceType: () => Le, HeaderFooterType: () => De, HeaderWrapper: () => Fi, HeadingLevel: () => Se, HeightRule: () => $s, HorizontalPosition: () => Ze, HorizontalPositionAlign: () => y, HorizontalPositionRelativeFrom: () => xe, HpsMeasureElement: () => R, HyperlinkType: () => Oe, IgnoreIfEmptyXmlComponent: () => i, ImageRun: () => qt, ImportDotx: () => wa, ImportedRootElementAttributes: () => p, ImportedXmlComponent: () => h, Indent: () => H, InitializableXmlComponent: () => f, InsertedTextRun: () => Jo, InternalHyperlink: () => Ir, LeaderType: () => Re, Level: () => Xi, LevelBase: () => $i, LevelForOverride: () => qi, LevelFormat: () => Ui, LevelOverride: () => no, LevelSuffix: () => Hi, LineNumberAttributes: () => Gr, LineNumberRestartFormat: () => Fe, LineNumberType: () => Vr, LineRuleType: () => Ae, Math: () => Rn, MathAccentCharacter: () => Pn, MathAngledBrackets: () => bs, MathBase: () => Fn, MathCurlyBrackets: () => ys, MathDegree: () => ns, MathDenominator: () => On, MathFraction: () => Ln, MathFunction: () => ls, MathFunctionName: () => cs, MathFunctionProperties: () => us, MathIntegral: () => $n, MathLimitLocation: () => Mn, MathNAryProperties: () => Wn, MathNumerator: () => kn, MathPreSubSuperScript: () => ts, MathPreSubSuperScriptProperties: () => es, MathRadical: () => as, MathRadicalProperties: () => os, MathRoundBrackets: () => ws, MathRun: () => Cn, MathSquareBrackets: () => gs, MathSubScript: () => Yn, MathSubScriptElement: () => Kn, MathSubScriptProperties: () => Zn, MathSubSuperScript: () => Qn, MathSubSuperScriptProperties: () => Jn, MathSum: () => Vn, MathSuperScript: () => qn, MathSuperScriptElement: () => Gn, MathSuperScriptProperties: () => Xn, Media: () => Bi, NumberFormat: () => x, NumberProperties: () => xr, NumberValueElement: () => C, Numbering: () => oo, OnOffElement: () => I, OutlineLevel: () => Dr, OverlapType: () => Gs, Packer: () => pa, PageBorderDisplay: () => Be, PageBorderOffsetFrom: () => Me, PageBorderZOrder: () => Ue, PageBorders: () => Xr, PageBreak: () => hr, PageBreakBefore: () => dr, PageMargin: () => Zr, PageMarginAttributes: () => qr, PageNumber: () => A, PageNumberSeparator: () => He, PageNumberType: () => Jr, PageNumberTypeAttributes: () => Yr, PageOrientation: () => ze, PageReference: () => Fr, PageSize: () => en, PageSizeAttributes: () => Qr, PageTextDirection: () => rn, PageTextDirectionType: () => je, Paragraph: () => In, ParagraphProperties: () => Sn, ParagraphPropertiesDefaults: () => Po, PrettifyType: () => la, RelativeHorizontalPosition: () => Ws, RelativeVerticalPosition: () => Ks, Run: () => me, RunFonts: () => ae, RunProperties: () => pe, RunPropertiesChange: () => de, RunPropertiesDefaults: () => Fo, SectionProperties: () => cn, SectionType: () => We, SectionTypeAttributes: () => nn, SequentialIdentifier: () => Yt, Shading: () => Y, ShadingType: () => _, SimpleField: () => er, SimpleMailMergeField: () => tr, SimplePos: () => Ve, SpaceType: () => v, Spacing: () => mr, StringContainer: () => k, StringEnumValueElement: () => O, StringValueElement: () => N, Style: () => wr, StyleForCharacter: () => bo, StyleForParagraph: () => yo, StyleLevel: () => Vo, Styles: () => Do, SymbolRun: () => be, TDirection: () => Fs, Tab: () => Jt, TabAttributes: () => yr, TabStop: () => gr, TabStopItem: () => br, TabStopPosition: () => Ne, TabStopType: () => Ie, Table: () => ei, TableAnchorType: () => js, TableBorders: () => zs, TableCell: () => Ms, TableCellBorders: () => Cs, TableFloatOptionsAttributes: () => qs, TableFloatProperties: () => Zs, TableLayout: () => Js, TableLayoutType: () => Vs, TableOfContents: () => Go, TableProperties: () => Qs, TableRow: () => si, TableRowHeight: () => ri, TableRowHeightAttributes: () => ti, TableRowProperties: () => ni, TableWidthElement: () => Rs, TextDirection: () => Ss, TextRun: () => we, TextWrappingSide: () => Ee, TextWrappingType: () => _e, ThematicBreak: () => M, Type: () => sn, Underline: () => he, UnderlineType: () => T, VerticalAlign: () => ke, VerticalAlignAttributes: () => Br, VerticalAlignElement: () => Mr, VerticalMerge: () => Ds, VerticalMergeType: () => As, VerticalPosition: () => Je, VerticalPositionAlign: () => b, VerticalPositionRelativeFrom: () => ve, WORKAROUND: () => zo, WORKAROUND2: () => Mi, WORKAROUND3: () => d, WORKAROUND4: () => rs, WidthType: () => Es, WrapNone: () => Rt, WrapSquare: () => Ct, WrapTight: () => kt, WrapTopAndBottom: () => Dt, XmlAttributeComponent: () => o, XmlComponent: () => s, convertInchesToTwip: () => ga.convertInchesToTwip, convertMillimetersToTwip: () => ga.convertMillimetersToTwip, convertToXmlComponent: () => u, dateTimeValue: () => ga.dateTimeValue, decimalNumber: () => ga.decimalNumber, eighthPointMeasureValue: () => ga.eighthPointMeasureValue, hexColorValue: () => ga.hexColorValue, hpsMeasureValue: () => ga.hpsMeasureValue, longHexNumber: () => ga.longHexNumber, measurementOrPercentValue: () => ga.measurementOrPercentValue, percentageValue: () => ga.percentageValue, pointMeasureValue: () => ga.pointMeasureValue, positiveUniversalMeasureValue: () => ga.positiveUniversalMeasureValue, sectionMarginDefaults: () => on, sectionPageSizeDefaults: () => an, shortHexNumber: () => ga.shortHexNumber, signedHpsMeasureValue: () => ga.signedHpsMeasureValue, signedTwipsMeasureValue: () => ga.signedTwipsMeasureValue, twipsMeasureValue: () => ga.twipsMeasureValue, uCharHexNumber: () => ga.uCharHexNumber, uniqueId: () => ga.uniqueId, uniqueNumericId: () => ga.uniqueNumericId, universalMeasureValue: () => ga.universalMeasureValue, unsignedDecimalNumber: () => ga.unsignedDecimalNumber });
        class e2 {
          constructor(e3) {
            this.rootKey = e3;
          }
        }
        const t2 = Object.seal({});
        class s extends e2 {
          constructor(e3) {
            super(e3), this.root = new Array();
          }
          prepForXml(r2) {
            var n2;
            const s2 = this.root.map((t3) => t3 instanceof e2 ? t3.prepForXml(r2) : t3).filter((e3) => void 0 !== e3);
            return { [this.rootKey]: s2.length ? 1 === s2.length && (null === (n2 = s2[0]) || void 0 === n2 ? void 0 : n2._attr) ? s2[0] : s2 : t2 };
          }
          addChildElement(e3) {
            return this.root.push(e3), this;
          }
        }
        class i extends s {
          prepForXml(e3) {
            const t3 = super.prepForXml(e3);
            if (t3 && ("object" != typeof t3[this.rootKey] || Object.keys(t3[this.rootKey]).length))
              return t3;
          }
        }
        class o extends e2 {
          constructor(e3) {
            super("_attr"), this.root = e3;
          }
          prepForXml(e3) {
            const t3 = {};
            return Object.keys(this.root).forEach((e4) => {
              const r2 = this.root[e4];
              if (void 0 !== r2) {
                const n2 = this.xmlKeys && this.xmlKeys[e4] || e4;
                t3[n2] = r2;
              }
            }), { _attr: t3 };
          }
        }
        class a extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { val: "w:val", color: "w:color", fill: "w:fill", space: "w:space", sz: "w:sz", type: "w:type", rsidR: "w:rsidR", rsidRPr: "w:rsidRPr", rsidSect: "w:rsidSect", w: "w:w", h: "w:h", top: "w:top", right: "w:right", bottom: "w:bottom", left: "w:left", header: "w:header", footer: "w:footer", gutter: "w:gutter", linePitch: "w:linePitch", pos: "w:pos" };
          }
        }
        var c = r(7888);
        const u = (e3) => {
          switch (e3.type) {
            case void 0:
            case "element":
              const t3 = new h(e3.name, e3.attributes), r2 = e3.elements || [];
              for (const e4 of r2) {
                const r3 = u(e4);
                void 0 !== r3 && t3.push(r3);
              }
              return t3;
            case "text":
              return e3.text;
            default:
              return;
          }
        };
        class l extends o {
        }
        class h extends s {
          static fromXmlString(e3) {
            const t3 = (0, c.xml2js)(e3, { compact: false });
            return u(t3);
          }
          constructor(e3, t3) {
            super(e3), t3 && this.root.push(new l(t3));
          }
          push(e3) {
            this.root.push(e3);
          }
        }
        class p extends s {
          constructor(e3) {
            super(""), this._attr = e3;
          }
          prepForXml(e3) {
            return { _attr: this._attr };
          }
        }
        const d = "";
        class f extends s {
          constructor(e3, t3) {
            super(e3), t3 && (this.root = t3.root);
          }
        }
        var m, w, g, y, b, x, v, _, E, T, A, S = r(6595);
        class I extends s {
          constructor(e3, t3 = true) {
            super(e3), true !== t3 && this.root.push(new a({ val: t3 }));
          }
        }
        class R extends s {
          constructor(e3, t3) {
            super(e3), this.root.push(new a({ val: (0, S.KR)(t3) }));
          }
        }
        class N extends s {
          constructor(e3, t3) {
            super(e3), this.root.push(new a({ val: t3 }));
          }
        }
        class C extends s {
          constructor(e3, t3) {
            super(e3), this.root.push(new a({ val: t3 }));
          }
        }
        class O extends s {
          constructor(e3, t3) {
            super(e3), this.root.push(new a({ val: t3 }));
          }
        }
        class k extends s {
          constructor(e3, t3) {
            super(e3), this.root.push(t3);
          }
        }
        !function(e3) {
          e3.START = "start", e3.CENTER = "center", e3.END = "end", e3.BOTH = "both", e3.MEDIUM_KASHIDA = "mediumKashida", e3.DISTRIBUTE = "distribute", e3.NUM_TAB = "numTab", e3.HIGH_KASHIDA = "highKashida", e3.LOW_KASHIDA = "lowKashida", e3.THAI_DISTRIBUTE = "thaiDistribute", e3.LEFT = "left", e3.RIGHT = "right", e3.JUSTIFIED = "both";
        }(m || (m = {}));
        class L extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { val: "w:val" };
          }
        }
        class D extends s {
          constructor(e3) {
            super("w:jc"), this.root.push(new L({ val: e3 }));
          }
        }
        class P extends s {
          constructor(e3, { color: t3, size: r2, space: n2, style: s2 }) {
            super(e3), this.root.push(new F({ style: s2, color: void 0 === t3 ? void 0 : (0, S.dg)(t3), size: void 0 === r2 ? void 0 : (0, S.LV)(r2), space: void 0 === n2 ? void 0 : (0, S.gg)(n2) }));
          }
        }
        class F extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { style: "w:val", color: "w:color", size: "w:sz", space: "w:space" };
          }
        }
        !function(e3) {
          e3.SINGLE = "single", e3.DASH_DOT_STROKED = "dashDotStroked", e3.DASHED = "dashed", e3.DASH_SMALL_GAP = "dashSmallGap", e3.DOT_DASH = "dotDash", e3.DOT_DOT_DASH = "dotDotDash", e3.DOTTED = "dotted", e3.DOUBLE = "double", e3.DOUBLE_WAVE = "doubleWave", e3.INSET = "inset", e3.NIL = "nil", e3.NONE = "none", e3.OUTSET = "outset", e3.THICK = "thick", e3.THICK_THIN_LARGE_GAP = "thickThinLargeGap", e3.THICK_THIN_MEDIUM_GAP = "thickThinMediumGap", e3.THICK_THIN_SMALL_GAP = "thickThinSmallGap", e3.THIN_THICK_LARGE_GAP = "thinThickLargeGap", e3.THIN_THICK_MEDIUM_GAP = "thinThickMediumGap", e3.THIN_THICK_SMALL_GAP = "thinThickSmallGap", e3.THIN_THICK_THIN_LARGE_GAP = "thinThickThinLargeGap", e3.THIN_THICK_THIN_MEDIUM_GAP = "thinThickThinMediumGap", e3.THIN_THICK_THIN_SMALL_GAP = "thinThickThinSmallGap", e3.THREE_D_EMBOSS = "threeDEmboss", e3.THREE_D_ENGRAVE = "threeDEngrave", e3.TRIPLE = "triple", e3.WAVE = "wave";
        }(w || (w = {}));
        class B extends i {
          constructor(e3) {
            super("w:pBdr"), e3.top && this.root.push(new P("w:top", e3.top)), e3.bottom && this.root.push(new P("w:bottom", e3.bottom)), e3.left && this.root.push(new P("w:left", e3.left)), e3.right && this.root.push(new P("w:right", e3.right));
          }
        }
        class M extends s {
          constructor() {
            super("w:pBdr");
            const e3 = new P("w:bottom", { color: "auto", space: 1, style: w.SINGLE, size: 6 });
            this.root.push(e3);
          }
        }
        class U extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { start: "w:start", end: "w:end", left: "w:left", right: "w:right", hanging: "w:hanging", firstLine: "w:firstLine" };
          }
        }
        class H extends s {
          constructor({ start: e3, end: t3, left: r2, right: n2, hanging: s2, firstLine: i2 }) {
            super("w:ind"), this.root.push(new U({ start: void 0 === e3 ? void 0 : (0, S.xb)(e3), end: void 0 === t3 ? void 0 : (0, S.xb)(t3), left: void 0 === r2 ? void 0 : (0, S.xb)(r2), right: void 0 === n2 ? void 0 : (0, S.xb)(n2), hanging: void 0 === s2 ? void 0 : (0, S.Jd)(s2), firstLine: void 0 === i2 ? void 0 : (0, S.Jd)(i2) }));
          }
        }
        class z extends s {
          constructor() {
            super("w:br");
          }
        }
        !function(e3) {
          e3.BEGIN = "begin", e3.END = "end", e3.SEPARATE = "separate";
        }(g || (g = {}));
        class j extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { type: "w:fldCharType", dirty: "w:dirty" };
          }
        }
        class W extends s {
          constructor(e3) {
            super("w:fldChar"), this.root.push(new j({ type: g.BEGIN, dirty: e3 }));
          }
        }
        class K extends s {
          constructor(e3) {
            super("w:fldChar"), this.root.push(new j({ type: g.SEPARATE, dirty: e3 }));
          }
        }
        class G extends s {
          constructor(e3) {
            super("w:fldChar"), this.root.push(new j({ type: g.END, dirty: e3 }));
          }
        }
        !function(e3) {
          e3.CENTER = "center", e3.INSIDE = "inside", e3.LEFT = "left", e3.OUTSIDE = "outside", e3.RIGHT = "right";
        }(y || (y = {})), function(e3) {
          e3.BOTTOM = "bottom", e3.CENTER = "center", e3.INSIDE = "inside", e3.OUTSIDE = "outside", e3.TOP = "top";
        }(b || (b = {})), function(e3) {
          e3.DECIMAL = "decimal", e3.UPPER_ROMAN = "upperRoman", e3.LOWER_ROMAN = "lowerRoman", e3.UPPER_LETTER = "upperLetter", e3.LOWER_LETTER = "lowerLetter", e3.ORDINAL = "ordinal", e3.CARDINAL_TEXT = "cardinalText", e3.ORDINAL_TEXT = "ordinalText", e3.HEX = "hex", e3.CHICAGO = "chicago", e3.IDEOGRAPH_DIGITAL = "ideographDigital", e3.JAPANESE_COUNTING = "japaneseCounting", e3.AIUEO = "aiueo", e3.IROHA = "iroha", e3.DECIMAL_FULL_WIDTH = "decimalFullWidth", e3.DECIMAL_HALF_WIDTH = "decimalHalfWidth", e3.JAPANESE_LEGAL = "japaneseLegal", e3.JAPANESE_DIGITAL_TEN_THOUSAND = "japaneseDigitalTenThousand", e3.DECIMAL_ENCLOSED_CIRCLE = "decimalEnclosedCircle", e3.DECIMAL_FULL_WIDTH_2 = "decimalFullWidth2", e3.AIUEO_FULL_WIDTH = "aiueoFullWidth", e3.IROHA_FULL_WIDTH = "irohaFullWidth", e3.DECIMAL_ZERO = "decimalZero", e3.BULLET = "bullet", e3.GANADA = "ganada", e3.CHOSUNG = "chosung", e3.DECIMAL_ENCLOSED_FULL_STOP = "decimalEnclosedFullstop", e3.DECIMAL_ENCLOSED_PAREN = "decimalEnclosedParen", e3.DECIMAL_ENCLOSED_CIRCLE_CHINESE = "decimalEnclosedCircleChinese", e3.IDEOGRAPH_ENCLOSED_CIRCLE = "ideographEnclosedCircle", e3.IDEOGRAPH_TRADITIONAL = "ideographTraditional", e3.IDEOGRAPH_ZODIAC = "ideographZodiac", e3.IDEOGRAPH_ZODIAC_TRADITIONAL = "ideographZodiacTraditional", e3.TAIWANESE_COUNTING = "taiwaneseCounting", e3.IDEOGRAPH_LEGAL_TRADITIONAL = "ideographLegalTraditional", e3.TAIWANESE_COUNTING_THOUSAND = "taiwaneseCountingThousand", e3.TAIWANESE_DIGITAL = "taiwaneseDigital", e3.CHINESE_COUNTING = "chineseCounting", e3.CHINESE_LEGAL_SIMPLIFIED = "chineseLegalSimplified", e3.CHINESE_COUNTING_TEN_THOUSAND = "chineseCountingThousand", e3.KOREAN_DIGITAL = "koreanDigital", e3.KOREAN_COUNTING = "koreanCounting", e3.KOREAN_LEGAL = "koreanLegal", e3.KOREAN_DIGITAL_2 = "koreanDigital2", e3.VIETNAMESE_COUNTING = "vietnameseCounting", e3.RUSSIAN_LOWER = "russianLower", e3.RUSSIAN_UPPER = "russianUpper", e3.NONE = "none", e3.NUMBER_IN_DASH = "numberInDash", e3.HEBREW_1 = "hebrew1", e3.HEBREW_2 = "hebrew2", e3.ARABIC_ALPHA = "arabicAlpha", e3.ARABIC_ABJAD = "arabicAbjad", e3.HINDI_VOWELS = "hindiVowels", e3.HINDI_CONSONANTS = "hindiConsonants", e3.HINDI_NUMBERS = "hindiNumbers", e3.HINDI_COUNTING = "hindiCounting", e3.THAI_LETTERS = "thaiLetters", e3.THAI_NUMBERS = "thaiNumbers", e3.THAI_COUNTING = "thaiCounting", e3.BAHT_TEXT = "bahtText", e3.DOLLAR_TEXT = "dollarText";
        }(x || (x = {})), function(e3) {
          e3.DEFAULT = "default", e3.PRESERVE = "preserve";
        }(v || (v = {}));
        class V extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { space: "xml:space" };
          }
        }
        class $ extends s {
          constructor() {
            super("w:instrText"), this.root.push(new V({ space: v.PRESERVE })), this.root.push("PAGE");
          }
        }
        class X extends s {
          constructor() {
            super("w:instrText"), this.root.push(new V({ space: v.PRESERVE })), this.root.push("NUMPAGES");
          }
        }
        class q extends s {
          constructor() {
            super("w:instrText"), this.root.push(new V({ space: v.PRESERVE })), this.root.push("SECTIONPAGES");
          }
        }
        class Z extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { fill: "w:fill", color: "w:color", type: "w:val" };
          }
        }
        class Y extends s {
          constructor({ fill: e3, color: t3, type: r2 }) {
            super("w:shd"), this.root.push(new Z({ fill: void 0 === e3 ? void 0 : (0, S.dg)(e3), color: void 0 === t3 ? void 0 : (0, S.dg)(t3), type: r2 }));
          }
        }
        !function(e3) {
          e3.CLEAR = "clear", e3.DIAGONAL_CROSS = "diagCross", e3.DIAGONAL_STRIPE = "diagStripe", e3.HORIZONTAL_CROSS = "horzCross", e3.HORIZONTAL_STRIPE = "horzStripe", e3.NIL = "nil", e3.PERCENT_5 = "pct5", e3.PERCENT_10 = "pct10", e3.PERCENT_12 = "pct12", e3.PERCENT_15 = "pct15", e3.PERCENT_20 = "pct20", e3.PERCENT_25 = "pct25", e3.PERCENT_30 = "pct30", e3.PERCENT_35 = "pct35", e3.PERCENT_37 = "pct37", e3.PERCENT_40 = "pct40", e3.PERCENT_45 = "pct45", e3.PERCENT_50 = "pct50", e3.PERCENT_55 = "pct55", e3.PERCENT_60 = "pct60", e3.PERCENT_62 = "pct62", e3.PERCENT_65 = "pct65", e3.PERCENT_70 = "pct70", e3.PERCENT_75 = "pct75", e3.PERCENT_80 = "pct80", e3.PERCENT_85 = "pct85", e3.PERCENT_87 = "pct87", e3.PERCENT_90 = "pct90", e3.PERCENT_95 = "pct95", e3.REVERSE_DIAGONAL_STRIPE = "reverseDiagStripe", e3.SOLID = "solid", e3.THIN_DIAGONAL_CROSS = "thinDiagCross", e3.THIN_DIAGONAL_STRIPE = "thinDiagStripe", e3.THIN_HORIZONTAL_CROSS = "thinHorzCross", e3.THIN_REVERSE_DIAGONAL_STRIPE = "thinReverseDiagStripe", e3.THIN_VERTICAL_STRIPE = "thinVertStripe", e3.VERTICAL_STRIPE = "vertStripe";
        }(_ || (_ = {}));
        class J extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { id: "w:id", author: "w:author", date: "w:date" };
          }
        }
        !function(e3) {
          e3.DOT = "dot";
        }(E || (E = {}));
        class Q extends s {
          constructor(e3) {
            super("w:em"), this.root.push(new a({ val: e3 }));
          }
        }
        class ee extends Q {
          constructor(e3 = E.DOT) {
            super(e3);
          }
        }
        class te extends Q {
          constructor() {
            super(E.DOT);
          }
        }
        class re extends s {
          constructor(e3) {
            super("w:spacing"), this.root.push(new a({ val: (0, S.xb)(e3) }));
          }
        }
        class ne extends s {
          constructor(e3) {
            super("w:color"), this.root.push(new a({ val: (0, S.dg)(e3) }));
          }
        }
        class se extends s {
          constructor(e3) {
            super("w:highlight"), this.root.push(new a({ val: e3 }));
          }
        }
        class ie extends s {
          constructor(e3) {
            super("w:highlightCs"), this.root.push(new a({ val: e3 }));
          }
        }
        class oe extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { ascii: "w:ascii", cs: "w:cs", eastAsia: "w:eastAsia", hAnsi: "w:hAnsi", hint: "w:hint" };
          }
        }
        class ae extends s {
          constructor(e3, t3) {
            if (super("w:rFonts"), "string" == typeof e3) {
              const r2 = e3;
              this.root.push(new oe({ ascii: r2, cs: r2, eastAsia: r2, hAnsi: r2, hint: t3 }));
            } else {
              const t4 = e3;
              this.root.push(new oe(t4));
            }
          }
        }
        class ce extends s {
          constructor(e3) {
            super("w:vertAlign"), this.root.push(new a({ val: e3 }));
          }
        }
        class ue extends ce {
          constructor() {
            super("superscript");
          }
        }
        class le extends ce {
          constructor() {
            super("subscript");
          }
        }
        !function(e3) {
          e3.SINGLE = "single", e3.WORDS = "words", e3.DOUBLE = "double", e3.THICK = "thick", e3.DOTTED = "dotted", e3.DOTTEDHEAVY = "dottedHeavy", e3.DASH = "dash", e3.DASHEDHEAVY = "dashedHeavy", e3.DASHLONG = "dashLong", e3.DASHLONGHEAVY = "dashLongHeavy", e3.DOTDASH = "dotDash", e3.DASHDOTHEAVY = "dashDotHeavy", e3.DOTDOTDASH = "dotDotDash", e3.DASHDOTDOTHEAVY = "dashDotDotHeavy", e3.WAVE = "wave", e3.WAVYHEAVY = "wavyHeavy", e3.WAVYDOUBLE = "wavyDouble", e3.NONE = "none";
        }(T || (T = {}));
        class he extends s {
          constructor(e3 = T.SINGLE, t3) {
            super("w:u"), this.root.push(new a({ val: e3, color: void 0 === t3 ? void 0 : (0, S.dg)(t3) }));
          }
        }
        class pe extends i {
          constructor(e3) {
            var t3, r2;
            if (super("w:rPr"), !e3)
              return;
            void 0 !== e3.bold && this.push(new I("w:b", e3.bold)), (void 0 === e3.boldComplexScript && void 0 !== e3.bold || e3.boldComplexScript) && this.push(new I("w:bCs", null !== (t3 = e3.boldComplexScript) && void 0 !== t3 ? t3 : e3.bold)), void 0 !== e3.italics && this.push(new I("w:i", e3.italics)), (void 0 === e3.italicsComplexScript && void 0 !== e3.italics || e3.italicsComplexScript) && this.push(new I("w:iCs", null !== (r2 = e3.italicsComplexScript) && void 0 !== r2 ? r2 : e3.italics)), e3.underline && this.push(new he(e3.underline.type, e3.underline.color)), e3.emphasisMark && this.push(new ee(e3.emphasisMark.type)), e3.color && this.push(new ne(e3.color)), void 0 !== e3.size && this.push(new R("w:sz", e3.size));
            const n2 = void 0 === e3.sizeComplexScript || true === e3.sizeComplexScript ? e3.size : e3.sizeComplexScript;
            n2 && this.push(new R("w:szCs", n2)), void 0 !== e3.rightToLeft && this.push(new I("w:rtl", e3.rightToLeft)), void 0 !== e3.smallCaps ? this.push(new I("w:smallCaps", e3.smallCaps)) : void 0 !== e3.allCaps && this.push(new I("w:caps", e3.allCaps)), void 0 !== e3.strike && this.push(new I("w:strike", e3.strike)), void 0 !== e3.doubleStrike && this.push(new I("w:dstrike", e3.doubleStrike)), e3.subScript && this.push(new le()), e3.superScript && this.push(new ue()), e3.style && this.push(new N("w:rStyle", e3.style)), e3.font && ("string" == typeof e3.font ? this.push(new ae(e3.font)) : "name" in e3.font ? this.push(new ae(e3.font.name, e3.font.hint)) : this.push(new ae(e3.font))), e3.highlight && this.push(new se(e3.highlight));
            const s2 = void 0 === e3.highlightComplexScript || true === e3.highlightComplexScript ? e3.highlight : e3.highlightComplexScript;
            s2 && this.push(new ie(s2)), e3.characterSpacing && this.push(new re(e3.characterSpacing)), void 0 !== e3.emboss && this.push(new I("w:emboss", e3.emboss)), void 0 !== e3.imprint && this.push(new I("w:imprint", e3.imprint)), e3.shading && this.push(new Y(e3.shading)), e3.revision && this.push(new de(e3.revision)), e3.border && this.push(new P("w:bdr", e3.border)), e3.vanish && this.push(new I("w:vanish", e3.vanish)), e3.specVanish && this.push(new I("w:specVanish", e3.vanish)), void 0 !== e3.scale && this.push(new C("w:w", e3.scale));
          }
          push(e3) {
            this.root.push(e3);
          }
        }
        class de extends s {
          constructor(e3) {
            super("w:rPrChange"), this.root.push(new J({ id: e3.id, author: e3.author, date: e3.date })), this.addChildElement(new pe(e3));
          }
        }
        class fe extends s {
          constructor(e3) {
            var t3;
            return super("w:t"), "string" == typeof e3 ? (this.root.push(new V({ space: v.PRESERVE })), this.root.push(e3), this) : (this.root.push(new V({ space: null !== (t3 = e3.space) && void 0 !== t3 ? t3 : v.DEFAULT })), this.root.push(e3.text), this);
          }
        }
        !function(e3) {
          e3.CURRENT = "CURRENT", e3.TOTAL_PAGES = "TOTAL_PAGES", e3.TOTAL_PAGES_IN_SECTION = "TOTAL_PAGES_IN_SECTION";
        }(A || (A = {}));
        class me extends s {
          constructor(e3) {
            if (super("w:r"), this.properties = new pe(e3), this.root.push(this.properties), e3.break)
              for (let t3 = 0; t3 < e3.break; t3++)
                this.root.push(new z());
            if (e3.children)
              for (const t3 of e3.children)
                if ("string" != typeof t3)
                  this.root.push(t3);
                else
                  switch (t3) {
                    case A.CURRENT:
                      this.root.push(new W()), this.root.push(new $()), this.root.push(new K()), this.root.push(new G());
                      break;
                    case A.TOTAL_PAGES:
                      this.root.push(new W()), this.root.push(new X()), this.root.push(new K()), this.root.push(new G());
                      break;
                    case A.TOTAL_PAGES_IN_SECTION:
                      this.root.push(new W()), this.root.push(new q()), this.root.push(new K()), this.root.push(new G());
                      break;
                    default:
                      this.root.push(new fe(t3));
                  }
            else
              e3.text && this.root.push(new fe(e3.text));
          }
        }
        class we extends me {
          constructor(e3) {
            if ("string" == typeof e3)
              return super({}), this.root.push(new fe(e3)), this;
            super(e3);
          }
        }
        class ge extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { char: "w:char", symbolfont: "w:font" };
          }
        }
        class ye extends s {
          constructor(e3 = "", t3 = "Wingdings") {
            super("w:sym"), this.root.push(new ge({ char: e3, symbolfont: t3 }));
          }
        }
        class be extends me {
          constructor(e3) {
            if ("string" == typeof e3)
              return super({}), this.root.push(new ye(e3)), this;
            super(e3), this.root.push(new ye(e3.char, e3.symbolfont));
          }
        }
        var xe, ve, _e, Ee, Te, Ae, Se, Ie, Re, Ne, Ce, Oe, ke, Le, De, Pe, Fe, Be, Me, Ue, He, ze, je, We, Ke = r(5457);
        !function(e3) {
          e3.CHARACTER = "character", e3.COLUMN = "column", e3.INSIDE_MARGIN = "insideMargin", e3.LEFT_MARGIN = "leftMargin", e3.MARGIN = "margin", e3.OUTSIDE_MARGIN = "outsideMargin", e3.PAGE = "page", e3.RIGHT_MARGIN = "rightMargin";
        }(xe || (xe = {})), function(e3) {
          e3.BOTTOM_MARGIN = "bottomMargin", e3.INSIDE_MARGIN = "insideMargin", e3.LINE = "line", e3.MARGIN = "margin", e3.OUTSIDE_MARGIN = "outsideMargin", e3.PAGE = "page", e3.PARAGRAPH = "paragraph", e3.TOP_MARGIN = "topMargin";
        }(ve || (ve = {}));
        class Ge extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { x: "x", y: "y" };
          }
        }
        class Ve extends s {
          constructor() {
            super("wp:simplePos"), this.root.push(new Ge({ x: 0, y: 0 }));
          }
        }
        class $e extends s {
          constructor(e3) {
            super("wp:align"), this.root.push(e3);
          }
        }
        class Xe extends s {
          constructor(e3) {
            super("wp:posOffset"), this.root.push(e3.toString());
          }
        }
        class qe extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { relativeFrom: "relativeFrom" };
          }
        }
        class Ze extends s {
          constructor(e3) {
            if (super("wp:positionH"), this.root.push(new qe({ relativeFrom: e3.relative || xe.PAGE })), e3.align)
              this.root.push(new $e(e3.align));
            else {
              if (void 0 === e3.offset)
                throw new Error("There is no configuration provided for floating position (Align or offset)");
              this.root.push(new Xe(e3.offset));
            }
          }
        }
        class Ye extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { relativeFrom: "relativeFrom" };
          }
        }
        class Je extends s {
          constructor(e3) {
            if (super("wp:positionV"), this.root.push(new Ye({ relativeFrom: e3.relative || ve.PAGE })), e3.align)
              this.root.push(new $e(e3.align));
            else {
              if (void 0 === e3.offset)
                throw new Error("There is no configuration provided for floating position (Align or offset)");
              this.root.push(new Xe(e3.offset));
            }
          }
        }
        class Qe extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { uri: "uri" };
          }
        }
        class et extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { embed: "r:embed", cstate: "cstate" };
          }
        }
        class tt extends s {
          constructor(e3) {
            super("a:blip"), this.root.push(new et({ embed: `rId{${e3.fileName}}`, cstate: "none" }));
          }
        }
        class rt extends s {
          constructor() {
            super("a:srcRect");
          }
        }
        class nt extends s {
          constructor() {
            super("a:fillRect");
          }
        }
        class st extends s {
          constructor() {
            super("a:stretch"), this.root.push(new nt());
          }
        }
        class it extends s {
          constructor(e3) {
            super("pic:blipFill"), this.root.push(new tt(e3)), this.root.push(new rt()), this.root.push(new st());
          }
        }
        class ot extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { noChangeAspect: "noChangeAspect", noChangeArrowheads: "noChangeArrowheads" };
          }
        }
        class at extends s {
          constructor() {
            super("a:picLocks"), this.root.push(new ot({ noChangeAspect: 1, noChangeArrowheads: 1 }));
          }
        }
        class ct extends s {
          constructor() {
            super("pic:cNvPicPr"), this.root.push(new at());
          }
        }
        class ut extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { id: "id", name: "name", descr: "descr" };
          }
        }
        class lt extends s {
          constructor() {
            super("pic:cNvPr"), this.root.push(new ut({ id: 0, name: "", descr: "" }));
          }
        }
        class ht extends s {
          constructor() {
            super("pic:nvPicPr"), this.root.push(new lt()), this.root.push(new ct());
          }
        }
        class pt extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { xmlns: "xmlns:pic" };
          }
        }
        class dt extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { cx: "cx", cy: "cy" };
          }
        }
        class ft extends s {
          constructor(e3, t3) {
            super("a:ext"), this.attributes = new dt({ cx: e3, cy: t3 }), this.root.push(this.attributes);
          }
        }
        class mt extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { x: "x", y: "y" };
          }
        }
        class wt extends s {
          constructor() {
            super("a:off"), this.root.push(new mt({ x: 0, y: 0 }));
          }
        }
        class gt extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { flipVertical: "flipV", flipHorizontal: "flipH", rotation: "rot" };
          }
        }
        class yt extends s {
          constructor(e3) {
            var t3, r2;
            super("a:xfrm"), this.root.push(new gt({ flipVertical: null === (t3 = e3.flip) || void 0 === t3 ? void 0 : t3.vertical, flipHorizontal: null === (r2 = e3.flip) || void 0 === r2 ? void 0 : r2.horizontal, rotation: e3.rotation })), this.extents = new ft(e3.emus.x, e3.emus.y), this.root.push(new wt()), this.root.push(this.extents);
          }
        }
        class bt extends s {
          constructor() {
            super("a:avLst");
          }
        }
        class xt extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { prst: "prst" };
          }
        }
        class vt extends s {
          constructor() {
            super("a:prstGeom"), this.root.push(new xt({ prst: "rect" })), this.root.push(new bt());
          }
        }
        class _t extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { bwMode: "bwMode" };
          }
        }
        class Et extends s {
          constructor(e3) {
            super("pic:spPr"), this.root.push(new _t({ bwMode: "auto" })), this.form = new yt(e3), this.root.push(this.form), this.root.push(new vt());
          }
        }
        class Tt extends s {
          constructor(e3, t3) {
            super("pic:pic"), this.root.push(new pt({ xmlns: "http://schemas.openxmlformats.org/drawingml/2006/picture" })), this.root.push(new ht()), this.root.push(new it(e3)), this.root.push(new Et(t3));
          }
        }
        class At extends s {
          constructor(e3, t3) {
            super("a:graphicData"), this.root.push(new Qe({ uri: "http://schemas.openxmlformats.org/drawingml/2006/picture" })), this.pic = new Tt(e3, t3), this.root.push(this.pic);
          }
        }
        class St extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { a: "xmlns:a" };
          }
        }
        class It extends s {
          constructor(e3, t3) {
            super("a:graphic"), this.root.push(new St({ a: "http://schemas.openxmlformats.org/drawingml/2006/main" })), this.data = new At(e3, t3), this.root.push(this.data);
          }
        }
        !function(e3) {
          e3[e3.NONE = 0] = "NONE", e3[e3.SQUARE = 1] = "SQUARE", e3[e3.TIGHT = 2] = "TIGHT", e3[e3.TOP_AND_BOTTOM = 3] = "TOP_AND_BOTTOM";
        }(_e || (_e = {})), function(e3) {
          e3.BOTH_SIDES = "bothSides", e3.LEFT = "left", e3.RIGHT = "right", e3.LARGEST = "largest";
        }(Ee || (Ee = {}));
        class Rt extends s {
          constructor() {
            super("wp:wrapNone");
          }
        }
        class Nt extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { distT: "distT", distB: "distB", distL: "distL", distR: "distR", wrapText: "wrapText" };
          }
        }
        class Ct extends s {
          constructor(e3, t3 = { top: 0, bottom: 0, left: 0, right: 0 }) {
            super("wp:wrapSquare"), this.root.push(new Nt({ wrapText: e3.side || Ee.BOTH_SIDES, distT: t3.top, distB: t3.bottom, distL: t3.left, distR: t3.right }));
          }
        }
        class Ot extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { distT: "distT", distB: "distB" };
          }
        }
        class kt extends s {
          constructor(e3 = { top: 0, bottom: 0 }) {
            super("wp:wrapTight"), this.root.push(new Ot({ distT: e3.top, distB: e3.bottom }));
          }
        }
        class Lt extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { distT: "distT", distB: "distB" };
          }
        }
        class Dt extends s {
          constructor(e3 = { top: 0, bottom: 0 }) {
            super("wp:wrapTopAndBottom"), this.root.push(new Lt({ distT: e3.top, distB: e3.bottom }));
          }
        }
        class Pt extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { id: "id", name: "name", description: "descr", title: "title" };
          }
        }
        class Ft extends s {
          constructor({ name: e3, description: t3, title: r2 } = { name: "", description: "", title: "" }) {
            super("wp:docPr"), this.root.push(new Pt({ id: (0, Ke.NY)(), name: e3, description: t3, title: r2 }));
          }
        }
        class Bt extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { b: "b", l: "l", r: "r", t: "t" };
          }
        }
        class Mt extends s {
          constructor() {
            super("wp:effectExtent"), this.root.push(new Bt({ b: 0, l: 0, r: 0, t: 0 }));
          }
        }
        class Ut extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { cx: "cx", cy: "cy" };
          }
        }
        class Ht extends s {
          constructor(e3, t3) {
            super("wp:extent"), this.attributes = new Ut({ cx: e3, cy: t3 }), this.root.push(this.attributes);
          }
        }
        class zt extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { xmlns: "xmlns:a", noChangeAspect: "noChangeAspect" };
          }
        }
        class jt extends s {
          constructor() {
            super("a:graphicFrameLocks"), this.root.push(new zt({ xmlns: "http://schemas.openxmlformats.org/drawingml/2006/main", noChangeAspect: 1 }));
          }
        }
        class Wt extends s {
          constructor() {
            super("wp:cNvGraphicFramePr"), this.root.push(new jt());
          }
        }
        class Kt extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { distT: "distT", distB: "distB", distL: "distL", distR: "distR", allowOverlap: "allowOverlap", behindDoc: "behindDoc", layoutInCell: "layoutInCell", locked: "locked", relativeHeight: "relativeHeight", simplePos: "simplePos" };
          }
        }
        class Gt extends s {
          constructor(e3, t3, r2) {
            super("wp:anchor");
            const n2 = Object.assign({ allowOverlap: true, behindDocument: false, lockAnchor: false, layoutInCell: true, verticalPosition: {}, horizontalPosition: {} }, r2.floating);
            if (this.root.push(new Kt({ distT: n2.margins && n2.margins.top || 0, distB: n2.margins && n2.margins.bottom || 0, distL: n2.margins && n2.margins.left || 0, distR: n2.margins && n2.margins.right || 0, simplePos: "0", allowOverlap: true === n2.allowOverlap ? "1" : "0", behindDoc: true === n2.behindDocument ? "1" : "0", locked: true === n2.lockAnchor ? "1" : "0", layoutInCell: true === n2.layoutInCell ? "1" : "0", relativeHeight: n2.zIndex ? n2.zIndex : t3.emus.y })), this.root.push(new Ve()), this.root.push(new Ze(n2.horizontalPosition)), this.root.push(new Je(n2.verticalPosition)), this.root.push(new Ht(t3.emus.x, t3.emus.y)), this.root.push(new Mt()), void 0 !== r2.floating && void 0 !== r2.floating.wrap)
              switch (r2.floating.wrap.type) {
                case _e.SQUARE:
                  this.root.push(new Ct(r2.floating.wrap, r2.floating.margins));
                  break;
                case _e.TIGHT:
                  this.root.push(new kt(r2.floating.margins));
                  break;
                case _e.TOP_AND_BOTTOM:
                  this.root.push(new Dt(r2.floating.margins));
                  break;
                case _e.NONE:
                default:
                  this.root.push(new Rt());
              }
            else
              this.root.push(new Rt());
            this.root.push(new Ft(r2.docProperties)), this.root.push(new Wt()), this.root.push(new It(e3, t3));
          }
        }
        class Vt extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { distT: "distT", distB: "distB", distL: "distL", distR: "distR" };
          }
        }
        class $t extends s {
          constructor({ mediaData: e3, transform: t3, docProperties: r2 }) {
            super("wp:inline"), this.root.push(new Vt({ distT: 0, distB: 0, distL: 0, distR: 0 })), this.extent = new Ht(t3.emus.x, t3.emus.y), this.graphic = new It(e3, t3), this.root.push(this.extent), this.root.push(new Mt()), this.root.push(new Ft(r2)), this.root.push(new Wt()), this.root.push(this.graphic);
          }
        }
        class Xt extends s {
          constructor(e3, t3 = {}) {
            super("w:drawing"), t3.floating ? this.root.push(new Gt(e3, e3.transformation, t3)) : (this.inline = new $t({ mediaData: e3, transform: e3.transformation, docProperties: t3.docProperties }), this.root.push(this.inline));
          }
        }
        class qt extends me {
          constructor(e3) {
            super({}), this.key = `${(0, Ke.EL)()}.png`;
            const t3 = "string" == typeof e3.data ? this.convertDataURIToBinary(e3.data) : e3.data;
            this.imageData = { stream: t3, fileName: this.key, transformation: { pixels: { x: Math.round(e3.transformation.width), y: Math.round(e3.transformation.height) }, emus: { x: Math.round(9525 * e3.transformation.width), y: Math.round(9525 * e3.transformation.height) }, flip: e3.transformation.flip, rotation: e3.transformation.rotation ? 6e4 * e3.transformation.rotation : void 0 } };
            const r2 = new Xt(this.imageData, { floating: e3.floating, docProperties: e3.altText });
            this.root.push(r2);
          }
          prepForXml(e3) {
            return e3.file.Media.addImage(this.key, this.imageData), super.prepForXml(e3);
          }
          convertDataURIToBinary(e3) {
            if ("function" == typeof atob) {
              const t3 = ";base64,", r2 = e3.indexOf(t3), n2 = -1 === r2 ? 0 : r2 + t3.length;
              return new Uint8Array(atob(e3.substring(n2)).split("").map((e4) => e4.charCodeAt(0)));
            }
            return new (r(8764)).Buffer(e3, "base64");
          }
        }
        class Zt extends s {
          constructor(e3) {
            super("w:instrText"), this.root.push(new V({ space: v.PRESERVE })), this.root.push(`SEQ ${e3}`);
          }
        }
        class Yt extends me {
          constructor(e3) {
            super({}), this.root.push(new W(true)), this.root.push(new Zt(e3)), this.root.push(new K()), this.root.push(new G());
          }
        }
        class Jt extends s {
          constructor() {
            super("w:tab");
          }
        }
        class Qt extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { instr: "w:instr" };
          }
        }
        class er extends s {
          constructor(e3, t3) {
            super("w:fldSimple"), this.root.push(new Qt({ instr: e3 })), void 0 !== t3 && this.root.push(new we(t3));
          }
        }
        class tr extends er {
          constructor(e3) {
            super(` MERGEFIELD ${e3} `, `\xAB${e3}\xBB`);
          }
        }
        class rr extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { id: "w:id", initials: "w:initials", author: "w:author", date: "w:date" };
          }
        }
        class nr extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { id: "w:id" };
          }
        }
        class sr extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { "xmlns:cx": "xmlns:cx", "xmlns:cx1": "xmlns:cx1", "xmlns:cx2": "xmlns:cx2", "xmlns:cx3": "xmlns:cx3", "xmlns:cx4": "xmlns:cx4", "xmlns:cx5": "xmlns:cx5", "xmlns:cx6": "xmlns:cx6", "xmlns:cx7": "xmlns:cx7", "xmlns:cx8": "xmlns:cx8", "xmlns:mc": "xmlns:mc", "xmlns:aink": "xmlns:aink", "xmlns:am3d": "xmlns:am3d", "xmlns:o": "xmlns:o", "xmlns:r": "xmlns:r", "xmlns:m": "xmlns:m", "xmlns:v": "xmlns:v", "xmlns:wp14": "xmlns:wp14", "xmlns:wp": "xmlns:wp", "xmlns:w10": "xmlns:w10", "xmlns:w": "xmlns:w", "xmlns:w14": "xmlns:w14", "xmlns:w15": "xmlns:w15", "xmlns:w16cex": "xmlns:w16cex", "xmlns:w16cid": "xmlns:w16cid", "xmlns:w16": "xmlns:w16", "xmlns:w16sdtdh": "xmlns:w16sdtdh", "xmlns:w16se": "xmlns:w16se", "xmlns:wpg": "xmlns:wpg", "xmlns:wpi": "xmlns:wpi", "xmlns:wne": "xmlns:wne", "xmlns:wps": "xmlns:wps" };
          }
        }
        class ir extends s {
          constructor(e3) {
            super("w:commentRangeStart"), this.root.push(new nr({ id: e3 }));
          }
        }
        class or extends s {
          constructor(e3) {
            super("w:commentRangeEnd"), this.root.push(new nr({ id: e3 }));
          }
        }
        class ar extends s {
          constructor(e3) {
            super("w:commentReference"), this.root.push(new nr({ id: e3 }));
          }
        }
        class cr extends s {
          constructor({ id: e3, initials: t3, author: r2, date: n2 = /* @__PURE__ */ new Date(), text: s2 }) {
            super("w:comment"), this.root.push(new rr({ id: e3, initials: t3, author: r2, date: n2.toISOString() })), this.root.push(new In({ children: [new we(s2)] }));
          }
        }
        class ur extends s {
          constructor({ children: e3 }) {
            super("w:comments"), this.root.push(new sr({ "xmlns:cx": "http://schemas.microsoft.com/office/drawing/2014/chartex", "xmlns:cx1": "http://schemas.microsoft.com/office/drawing/2015/9/8/chartex", "xmlns:cx2": "http://schemas.microsoft.com/office/drawing/2015/10/21/chartex", "xmlns:cx3": "http://schemas.microsoft.com/office/drawing/2016/5/9/chartex", "xmlns:cx4": "http://schemas.microsoft.com/office/drawing/2016/5/10/chartex", "xmlns:cx5": "http://schemas.microsoft.com/office/drawing/2016/5/11/chartex", "xmlns:cx6": "http://schemas.microsoft.com/office/drawing/2016/5/12/chartex", "xmlns:cx7": "http://schemas.microsoft.com/office/drawing/2016/5/13/chartex", "xmlns:cx8": "http://schemas.microsoft.com/office/drawing/2016/5/14/chartex", "xmlns:mc": "http://schemas.openxmlformats.org/markup-compatibility/2006", "xmlns:aink": "http://schemas.microsoft.com/office/drawing/2016/ink", "xmlns:am3d": "http://schemas.microsoft.com/office/drawing/2017/model3d", "xmlns:o": "urn:schemas-microsoft-com:office:office", "xmlns:r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships", "xmlns:m": "http://schemas.openxmlformats.org/officeDocument/2006/math", "xmlns:v": "urn:schemas-microsoft-com:vml", "xmlns:wp14": "http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing", "xmlns:wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing", "xmlns:w10": "urn:schemas-microsoft-com:office:word", "xmlns:w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main", "xmlns:w14": "http://schemas.microsoft.com/office/word/2010/wordml", "xmlns:w15": "http://schemas.microsoft.com/office/word/2012/wordml", "xmlns:w16cex": "http://schemas.microsoft.com/office/word/2018/wordml/cex", "xmlns:w16cid": "http://schemas.microsoft.com/office/word/2016/wordml/cid", "xmlns:w16": "http://schemas.microsoft.com/office/word/2018/wordml", "xmlns:w16sdtdh": "http://schemas.microsoft.com/office/word/2020/wordml/sdtdatahash", "xmlns:w16se": "http://schemas.microsoft.com/office/word/2015/wordml/symex", "xmlns:wpg": "http://schemas.microsoft.com/office/word/2010/wordprocessingGroup", "xmlns:wpi": "http://schemas.microsoft.com/office/word/2010/wordprocessingInk", "xmlns:wne": "http://schemas.microsoft.com/office/word/2006/wordml", "xmlns:wps": "http://schemas.microsoft.com/office/word/2010/wordprocessingShape" }));
            for (const t3 of e3)
              this.root.push(new cr(t3));
          }
        }
        !function(e3) {
          e3.COLUMN = "column", e3.PAGE = "page";
        }(Te || (Te = {}));
        class lr extends s {
          constructor(e3) {
            super("w:br"), this.root.push(new a({ type: e3 }));
          }
        }
        class hr extends me {
          constructor() {
            super({}), this.root.push(new lr(Te.PAGE));
          }
        }
        class pr extends me {
          constructor() {
            super({}), this.root.push(new lr(Te.COLUMN));
          }
        }
        class dr extends s {
          constructor() {
            super("w:pageBreakBefore");
          }
        }
        !function(e3) {
          e3.AT_LEAST = "atLeast", e3.EXACTLY = "exactly", e3.EXACT = "exact", e3.AUTO = "auto";
        }(Ae || (Ae = {}));
        class fr extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { after: "w:after", before: "w:before", line: "w:line", lineRule: "w:lineRule" };
          }
        }
        class mr extends s {
          constructor(e3) {
            super("w:spacing"), this.root.push(new fr(e3));
          }
        }
        !function(e3) {
          e3.HEADING_1 = "Heading1", e3.HEADING_2 = "Heading2", e3.HEADING_3 = "Heading3", e3.HEADING_4 = "Heading4", e3.HEADING_5 = "Heading5", e3.HEADING_6 = "Heading6", e3.TITLE = "Title";
        }(Se || (Se = {}));
        class wr extends s {
          constructor(e3) {
            super("w:pStyle"), this.root.push(new a({ val: e3 }));
          }
        }
        class gr extends s {
          constructor(e3) {
            super("w:tabs");
            for (const t3 of e3)
              this.root.push(new br(t3));
          }
        }
        !function(e3) {
          e3.LEFT = "left", e3.RIGHT = "right", e3.CENTER = "center", e3.BAR = "bar", e3.CLEAR = "clear", e3.DECIMAL = "decimal", e3.END = "end", e3.NUM = "num", e3.START = "start";
        }(Ie || (Ie = {})), function(e3) {
          e3.DOT = "dot", e3.HYPHEN = "hyphen", e3.MIDDLE_DOT = "middleDot", e3.NONE = "none", e3.UNDERSCORE = "underscore";
        }(Re || (Re = {})), function(e3) {
          e3[e3.MAX = 9026] = "MAX";
        }(Ne || (Ne = {}));
        class yr extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { val: "w:val", pos: "w:pos", leader: "w:leader" };
          }
        }
        class br extends s {
          constructor({ type: e3, position: t3, leader: r2 }) {
            super("w:tab"), this.root.push(new yr({ val: e3, pos: t3, leader: r2 }));
          }
        }
        class xr extends s {
          constructor(e3, t3) {
            super("w:numPr"), this.root.push(new vr(t3)), this.root.push(new _r(e3));
          }
        }
        class vr extends s {
          constructor(e3) {
            if (super("w:ilvl"), e3 > 9)
              throw new Error("Level cannot be greater than 9. Read more here: https://answers.microsoft.com/en-us/msoffice/forum/all/does-word-support-more-than-9-list-levels/d130fdcd-1781-446d-8c84-c6c79124e4d7");
            this.root.push(new a({ val: e3 }));
          }
        }
        class _r extends s {
          constructor(e3) {
            super("w:numId"), this.root.push(new a({ val: "string" == typeof e3 ? `{${e3}}` : e3 }));
          }
        }
        class Er extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { id: "Id", type: "Type", target: "Target", targetMode: "TargetMode" };
          }
        }
        !function(e3) {
          e3.EXTERNAL = "External";
        }(Ce || (Ce = {}));
        class Tr extends s {
          constructor(e3, t3, r2, n2) {
            super("Relationship"), this.root.push(new Er({ id: e3, type: t3, target: r2, targetMode: n2 }));
          }
        }
        class Ar extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { id: "r:id", history: "w:history", anchor: "w:anchor" };
          }
        }
        !function(e3) {
          e3.INTERNAL = "INTERNAL", e3.EXTERNAL = "EXTERNAL";
        }(Oe || (Oe = {}));
        class Sr extends s {
          constructor(e3, t3, r2) {
            super("w:hyperlink"), this.linkId = t3;
            const n2 = { history: 1, anchor: r2 || void 0, id: r2 ? void 0 : `rId${this.linkId}` }, s2 = new Ar(n2);
            this.root.push(s2), e3.forEach((e4) => {
              this.root.push(e4);
            });
          }
        }
        class Ir extends Sr {
          constructor(e3) {
            super(e3.children, (0, Ke.EL)(), e3.anchor);
          }
        }
        class Rr extends s {
          constructor(e3) {
            super("w:externalHyperlink"), this.options = e3;
          }
        }
        class Nr extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { id: "w:id", name: "w:name" };
          }
        }
        class Cr extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { id: "w:id" };
          }
        }
        class Or {
          constructor(e3) {
            const t3 = (0, Ke.NY)();
            this.start = new kr(e3.id, t3), this.children = e3.children, this.end = new Lr(t3);
          }
        }
        class kr extends s {
          constructor(e3, t3) {
            super("w:bookmarkStart");
            const r2 = new Nr({ name: e3, id: t3 });
            this.root.push(r2);
          }
        }
        class Lr extends s {
          constructor(e3) {
            super("w:bookmarkEnd");
            const t3 = new Cr({ id: e3 });
            this.root.push(t3);
          }
        }
        class Dr extends s {
          constructor(e3) {
            super("w:outlineLvl"), this.level = e3, this.root.push(new a({ val: e3 }));
          }
        }
        class Pr extends s {
          constructor(e3, t3 = {}) {
            super("w:instrText"), this.root.push(new V({ space: v.PRESERVE }));
            let r2 = `PAGEREF ${e3}`;
            t3.hyperlink && (r2 = `${r2} \\h`), t3.useRelativePosition && (r2 = `${r2} \\p`), this.root.push(r2);
          }
        }
        class Fr extends me {
          constructor(e3, t3 = {}) {
            super({ children: [new W(true), new Pr(e3, t3), new G()] });
          }
        }
        !function(e3) {
          e3.BOTTOM = "bottom", e3.CENTER = "center", e3.TOP = "top";
        }(ke || (ke = {}));
        class Br extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { verticalAlign: "w:val" };
          }
        }
        class Mr extends s {
          constructor(e3) {
            super("w:vAlign"), this.root.push(new Br({ verticalAlign: e3 }));
          }
        }
        !function(e3) {
          e3.DEFAULT = "default", e3.FIRST = "first", e3.EVEN = "even";
        }(Le || (Le = {}));
        class Ur extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { type: "w:type", id: "r:id" };
          }
        }
        !function(e3) {
          e3.HEADER = "w:headerReference", e3.FOOTER = "w:footerReference";
        }(De || (De = {}));
        class Hr extends s {
          constructor(e3, t3) {
            super(e3), this.root.push(new Ur({ type: t3.type || Le.DEFAULT, id: `rId${t3.id}` }));
          }
        }
        class zr extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { space: "w:space", count: "w:num", separate: "w:sep", equalWidth: "w:equalWidth" };
          }
        }
        class jr extends s {
          constructor({ space: e3, count: t3, separate: r2, equalWidth: n2, children: s2 }) {
            super("w:cols"), this.root.push(new zr({ space: void 0 === e3 ? void 0 : (0, S.Jd)(e3), count: void 0 === t3 ? void 0 : (0, S.vH)(t3), separate: r2, equalWidth: n2 })), !n2 && s2 && s2.forEach((e4) => this.addChildElement(e4));
          }
        }
        !function(e3) {
          e3.DEFAULT = "default", e3.LINES = "lines", e3.LINES_AND_CHARS = "linesAndChars", e3.SNAP_TO_CHARS = "snapToChars";
        }(Pe || (Pe = {}));
        class Wr extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { type: "w:type", linePitch: "w:linePitch", charSpace: "w:charSpace" };
          }
        }
        class Kr extends s {
          constructor(e3, t3, r2) {
            super("w:docGrid"), this.root.push(new Wr({ type: r2, linePitch: (0, S.vH)(e3), charSpace: t3 ? (0, S.vH)(t3) : void 0 }));
          }
        }
        !function(e3) {
          e3.NEW_PAGE = "newPage", e3.NEW_SECTION = "newSection", e3.CONTINUOUS = "continuous";
        }(Fe || (Fe = {}));
        class Gr extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { countBy: "w:countBy", start: "w:start", restart: "w:restart", distance: "w:distance" };
          }
        }
        class Vr extends s {
          constructor({ countBy: e3, start: t3, restart: r2, distance: n2 }) {
            super("w:lnNumType"), this.root.push(new Gr({ countBy: void 0 === e3 ? void 0 : (0, S.vH)(e3), start: void 0 === t3 ? void 0 : (0, S.vH)(t3), restart: r2, distance: void 0 === n2 ? void 0 : (0, S.Jd)(n2) }));
          }
        }
        !function(e3) {
          e3.ALL_PAGES = "allPages", e3.FIRST_PAGE = "firstPage", e3.NOT_FIRST_PAGE = "notFirstPage";
        }(Be || (Be = {})), function(e3) {
          e3.PAGE = "page", e3.TEXT = "text";
        }(Me || (Me = {})), function(e3) {
          e3.BACK = "back", e3.FRONT = "front";
        }(Ue || (Ue = {}));
        class $r extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { display: "w:display", offsetFrom: "w:offsetFrom", zOrder: "w:zOrder" };
          }
        }
        class Xr extends i {
          constructor(e3) {
            if (super("w:pgBorders"), !e3)
              return this;
            e3.pageBorders ? this.root.push(new $r({ display: e3.pageBorders.display, offsetFrom: e3.pageBorders.offsetFrom, zOrder: e3.pageBorders.zOrder })) : this.root.push(new $r({})), e3.pageBorderTop && this.root.push(new P("w:top", e3.pageBorderTop)), e3.pageBorderLeft && this.root.push(new P("w:left", e3.pageBorderLeft)), e3.pageBorderBottom && this.root.push(new P("w:bottom", e3.pageBorderBottom)), e3.pageBorderRight && this.root.push(new P("w:right", e3.pageBorderRight));
          }
        }
        class qr extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { top: "w:top", right: "w:right", bottom: "w:bottom", left: "w:left", header: "w:header", footer: "w:footer", gutter: "w:gutter" };
          }
        }
        class Zr extends s {
          constructor(e3, t3, r2, n2, s2, i2, o2) {
            super("w:pgMar"), this.root.push(new qr({ top: (0, S.xb)(e3), right: (0, S.Jd)(t3), bottom: (0, S.xb)(r2), left: (0, S.Jd)(n2), header: (0, S.Jd)(s2), footer: (0, S.Jd)(i2), gutter: (0, S.Jd)(o2) }));
          }
        }
        !function(e3) {
          e3.HYPHEN = "hyphen", e3.PERIOD = "period", e3.COLON = "colon", e3.EM_DASH = "emDash", e3.EN_DASH = "endash";
        }(He || (He = {}));
        class Yr extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { start: "w:start", formatType: "w:fmt", separator: "w:chapSep" };
          }
        }
        class Jr extends s {
          constructor({ start: e3, formatType: t3, separator: r2 }) {
            super("w:pgNumType"), this.root.push(new Yr({ start: void 0 === e3 ? void 0 : (0, S.vH)(e3), formatType: t3, separator: r2 }));
          }
        }
        !function(e3) {
          e3.PORTRAIT = "portrait", e3.LANDSCAPE = "landscape";
        }(ze || (ze = {}));
        class Qr extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { width: "w:w", height: "w:h", orientation: "w:orient" };
          }
        }
        class en extends s {
          constructor(e3, t3, r2) {
            super("w:pgSz");
            const n2 = r2 === ze.LANDSCAPE, s2 = (0, S.Jd)(e3), i2 = (0, S.Jd)(t3);
            this.root.push(new Qr({ width: n2 ? i2 : s2, height: n2 ? s2 : i2, orientation: r2 }));
          }
        }
        !function(e3) {
          e3.LEFT_TO_RIGHT_TOP_TO_BOTTOM = "lrTb", e3.TOP_TO_BOTTOM_RIGHT_TO_LEFT = "tbRl";
        }(je || (je = {}));
        class tn extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { val: "w:val" };
          }
        }
        class rn extends s {
          constructor(e3) {
            super("w:textDirection"), this.root.push(new tn({ val: e3 }));
          }
        }
        !function(e3) {
          e3.NEXT_PAGE = "nextPage", e3.NEXT_COLUMN = "nextColumn", e3.CONTINUOUS = "continuous", e3.EVEN_PAGE = "evenPage", e3.ODD_PAGE = "oddPage";
        }(We || (We = {}));
        class nn extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { val: "w:val" };
          }
        }
        class sn extends s {
          constructor(e3) {
            super("w:type"), this.root.push(new nn({ val: e3 }));
          }
        }
        const on = { TOP: "1in", RIGHT: "1in", BOTTOM: "1in", LEFT: "1in", HEADER: 708, FOOTER: 708, GUTTER: 0 }, an = { WIDTH: 11906, HEIGHT: 16838, ORIENTATION: ze.PORTRAIT };
        class cn extends s {
          constructor({ page: { size: { width: e3 = an.WIDTH, height: t3 = an.HEIGHT, orientation: r2 = an.ORIENTATION } = {}, margin: { top: n2 = on.TOP, right: s2 = on.RIGHT, bottom: i2 = on.BOTTOM, left: o2 = on.LEFT, header: a2 = on.HEADER, footer: c2 = on.FOOTER, gutter: u2 = on.GUTTER } = {}, pageNumbers: l2 = {}, borders: h2, textDirection: p2 } = {}, grid: { linePitch: d2 = 360, charSpace: f2, type: m2 } = {}, headerWrapperGroup: w2 = {}, footerWrapperGroup: g2 = {}, lineNumbers: y2, titlePage: b2, verticalAlign: x2, column: v2, type: _2 } = {}) {
            super("w:sectPr"), this.addHeaderFooterGroup(De.HEADER, w2), this.addHeaderFooterGroup(De.FOOTER, g2), _2 && this.root.push(new sn(_2)), this.root.push(new en(e3, t3, r2)), this.root.push(new Zr(n2, s2, i2, o2, a2, c2, u2)), h2 && this.root.push(new Xr(h2)), y2 && this.root.push(new Vr(y2)), this.root.push(new Jr(l2)), v2 && this.root.push(new jr(v2)), x2 && this.root.push(new Mr(x2)), void 0 !== b2 && this.root.push(new I("w:titlePg", b2)), p2 && this.root.push(new rn(p2)), this.root.push(new Kr(d2, f2, m2));
          }
          addHeaderFooterGroup(e3, t3) {
            t3.default && this.root.push(new Hr(e3, { type: Le.DEFAULT, id: t3.default.View.ReferenceId })), t3.first && this.root.push(new Hr(e3, { type: Le.FIRST, id: t3.first.View.ReferenceId })), t3.even && this.root.push(new Hr(e3, { type: Le.EVEN, id: t3.even.View.ReferenceId }));
          }
        }
        class un extends s {
          constructor() {
            super("w:body"), this.sections = [];
          }
          addSection(e3) {
            const t3 = this.sections.pop();
            this.root.push(this.createSectionParagraph(t3)), this.sections.push(new cn(e3));
          }
          prepForXml(e3) {
            return 1 === this.sections.length && (this.root.splice(0, 1), this.root.push(this.sections.pop())), super.prepForXml(e3);
          }
          push(e3) {
            this.root.push(e3);
          }
          createSectionParagraph(e3) {
            const t3 = new In({}), r2 = new Sn({});
            return r2.push(e3), t3.addChildElement(r2), t3;
          }
        }
        class ln extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { width: "w:w", space: "w:space" };
          }
        }
        class hn extends s {
          constructor({ width: e3, space: t3 }) {
            super("w:col"), this.root.push(new ln({ width: (0, S.Jd)(e3), space: void 0 === t3 ? void 0 : (0, S.Jd)(t3) }));
          }
        }
        class pn extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { wpc: "xmlns:wpc", mc: "xmlns:mc", o: "xmlns:o", r: "xmlns:r", m: "xmlns:m", v: "xmlns:v", wp14: "xmlns:wp14", wp: "xmlns:wp", w10: "xmlns:w10", w: "xmlns:w", w14: "xmlns:w14", w15: "xmlns:w15", wpg: "xmlns:wpg", wpi: "xmlns:wpi", wne: "xmlns:wne", wps: "xmlns:wps", Ignorable: "mc:Ignorable", cp: "xmlns:cp", dc: "xmlns:dc", dcterms: "xmlns:dcterms", dcmitype: "xmlns:dcmitype", xsi: "xmlns:xsi", type: "xsi:type", cx: "xmlns:cx", cx1: "xmlns:cx1", cx2: "xmlns:cx2", cx3: "xmlns:cx3", cx4: "xmlns:cx4", cx5: "xmlns:cx5", cx6: "xmlns:cx6", cx7: "xmlns:cx7", cx8: "xmlns:cx8", aink: "xmlns:aink", am3d: "xmlns:am3d", w16cex: "xmlns:w16cex", w16cid: "xmlns:w16cid", w16: "xmlns:w16", w16sdtdh: "xmlns:w16sdtdh", w16se: "xmlns:w16se" };
          }
        }
        class dn extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { color: "w:color", themeColor: "w:themeColor", themeShade: "w:themeShade", themeTint: "w:themeTint" };
          }
        }
        class fn extends s {
          constructor(e3) {
            super("w:background"), this.root.push(new dn({ color: void 0 === e3.color ? void 0 : (0, S.dg)(e3.color), themeColor: e3.themeColor, themeShade: void 0 === e3.themeShade ? void 0 : (0, S.xD)(e3.themeShade), themeTint: void 0 === e3.themeTint ? void 0 : (0, S.xD)(e3.themeTint) }));
          }
        }
        class mn extends s {
          constructor(e3) {
            super("w:document"), this.root.push(new pn({ wpc: "http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas", mc: "http://schemas.openxmlformats.org/markup-compatibility/2006", o: "urn:schemas-microsoft-com:office:office", r: "http://schemas.openxmlformats.org/officeDocument/2006/relationships", m: "http://schemas.openxmlformats.org/officeDocument/2006/math", v: "urn:schemas-microsoft-com:vml", wp14: "http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing", wp: "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing", w10: "urn:schemas-microsoft-com:office:word", w: "http://schemas.openxmlformats.org/wordprocessingml/2006/main", w14: "http://schemas.microsoft.com/office/word/2010/wordml", w15: "http://schemas.microsoft.com/office/word/2012/wordml", wpg: "http://schemas.microsoft.com/office/word/2010/wordprocessingGroup", wpi: "http://schemas.microsoft.com/office/word/2010/wordprocessingInk", wne: "http://schemas.microsoft.com/office/word/2006/wordml", wps: "http://schemas.microsoft.com/office/word/2010/wordprocessingShape", cx: "http://schemas.microsoft.com/office/drawing/2014/chartex", cx1: "http://schemas.microsoft.com/office/drawing/2015/9/8/chartex", cx2: "http://schemas.microsoft.com/office/drawing/2015/10/21/chartex", cx3: "http://schemas.microsoft.com/office/drawing/2016/5/9/chartex", cx4: "http://schemas.microsoft.com/office/drawing/2016/5/10/chartex", cx5: "http://schemas.microsoft.com/office/drawing/2016/5/11/chartex", cx6: "http://schemas.microsoft.com/office/drawing/2016/5/12/chartex", cx7: "http://schemas.microsoft.com/office/drawing/2016/5/13/chartex", cx8: "http://schemas.microsoft.com/office/drawing/2016/5/14/chartex", aink: "http://schemas.microsoft.com/office/drawing/2016/ink", am3d: "http://schemas.microsoft.com/office/drawing/2017/model3d", w16cex: "http://schemas.microsoft.com/office/word/2018/wordml/cex", w16cid: "http://schemas.microsoft.com/office/word/2016/wordml/cid", w16: "http://schemas.microsoft.com/office/word/2018/wordml", w16sdtdh: "http://schemas.microsoft.com/office/word/2020/wordml/sdtdatahash", w16se: "http://schemas.microsoft.com/office/word/2015/wordml/symex", Ignorable: "w14 w15 wp14" })), this.body = new un(), e3.background && this.root.push(new fn(e3.background)), this.root.push(this.body);
          }
          add(e3) {
            return this.body.push(e3), this;
          }
          get Body() {
            return this.body;
          }
        }
        class wn extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { xmlns: "xmlns" };
          }
        }
        class gn extends s {
          constructor() {
            super("Relationships"), this.root.push(new wn({ xmlns: "http://schemas.openxmlformats.org/package/2006/relationships" }));
          }
          addRelationship(e3) {
            this.root.push(e3);
          }
          createRelationship(e3, t3, r2, n2) {
            const s2 = new Tr(`rId${e3}`, t3, r2, n2);
            return this.addRelationship(s2), s2;
          }
          get RelationshipCount() {
            return this.root.length - 1;
          }
        }
        class yn {
          constructor(e3) {
            this.document = new mn(e3), this.relationships = new gn();
          }
          get View() {
            return this.document;
          }
          get Relationships() {
            return this.relationships;
          }
        }
        class bn extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { val: "w:val" };
          }
        }
        class xn extends s {
          constructor() {
            super("w:wordWrap"), this.root.push(new bn({ val: 0 }));
          }
        }
        var vn, _n, En;
        !function(e3) {
          e3.NONE = "none", e3.DROP = "drop", e3.MARGIN = "margin";
        }(vn || (vn = {})), function(e3) {
          e3.MARGIN = "margin", e3.PAGE = "page", e3.TEXT = "text";
        }(_n || (_n = {})), function(e3) {
          e3.AROUND = "around", e3.AUTO = "auto", e3.NONE = "none", e3.NOT_BESIDE = "notBeside", e3.THROUGH = "through", e3.TIGHT = "tight";
        }(En || (En = {}));
        class Tn extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { anchorLock: "w:anchorLock", dropCap: "w:dropCap", width: "w:w", height: "w:h", x: "w:x", y: "w:y", anchorHorizontal: "w:hAnchor", anchorVertical: "w:vAnchor", spaceHorizontal: "w:hSpace", spaceVertical: "w:vSpace", rule: "w:hRule", alignmentX: "w:xAlign", alignmentY: "w:yAlign", lines: "w:lines", wrap: "w:wrap" };
          }
        }
        class An extends s {
          constructor(e3) {
            var t3, r2;
            super("w:framePr"), this.root.push(new Tn({ anchorLock: e3.anchorLock, dropCap: e3.dropCap, width: e3.width, height: e3.height, x: e3.position ? e3.position.x : void 0, y: e3.position ? e3.position.y : void 0, anchorHorizontal: e3.anchor.horizontal, anchorVertical: e3.anchor.vertical, spaceHorizontal: null === (t3 = e3.space) || void 0 === t3 ? void 0 : t3.horizontal, spaceVertical: null === (r2 = e3.space) || void 0 === r2 ? void 0 : r2.vertical, rule: e3.rule, alignmentX: e3.alignment ? e3.alignment.x : void 0, alignmentY: e3.alignment ? e3.alignment.y : void 0, lines: e3.lines, wrap: e3.wrap }));
          }
        }
        class Sn extends i {
          constructor(e3) {
            var t3, r2;
            if (super("w:pPr"), this.numberingReferences = [], !e3)
              return this;
            e3.heading && this.push(new wr(e3.heading)), e3.bullet && this.push(new wr("ListParagraph")), e3.numbering && (e3.style || e3.heading || e3.numbering.custom || this.push(new wr("ListParagraph"))), e3.style && this.push(new wr(e3.style)), void 0 !== e3.keepNext && this.push(new I("w:keepNext", e3.keepNext)), void 0 !== e3.keepLines && this.push(new I("w:keepLines", e3.keepLines)), e3.pageBreakBefore && this.push(new dr()), e3.frame && this.push(new An(e3.frame)), void 0 !== e3.widowControl && this.push(new I("w:widowControl", e3.widowControl)), e3.bullet && this.push(new xr(1, e3.bullet.level)), e3.numbering && (this.numberingReferences.push({ reference: e3.numbering.reference, instance: null !== (t3 = e3.numbering.instance) && void 0 !== t3 ? t3 : 0 }), this.push(new xr(`${e3.numbering.reference}-${null !== (r2 = e3.numbering.instance) && void 0 !== r2 ? r2 : 0}`, e3.numbering.level))), e3.border && this.push(new B(e3.border)), e3.thematicBreak && this.push(new M()), e3.shading && this.push(new Y(e3.shading)), e3.wordWrap && this.push(new xn());
            const n2 = [...e3.rightTabStop ? [{ type: Ie.RIGHT, position: e3.rightTabStop }] : [], ...e3.tabStops ? e3.tabStops : [], ...e3.leftTabStop ? [{ type: Ie.LEFT, position: e3.leftTabStop }] : []];
            n2.length > 0 && this.push(new gr(n2)), void 0 !== e3.bidirectional && this.push(new I("w:bidi", e3.bidirectional)), e3.spacing && this.push(new mr(e3.spacing)), e3.indent && this.push(new H(e3.indent)), void 0 !== e3.contextualSpacing && this.push(new I("w:contextualSpacing", e3.contextualSpacing)), e3.alignment && this.push(new D(e3.alignment)), void 0 !== e3.outlineLevel && this.push(new Dr(e3.outlineLevel)), void 0 !== e3.suppressLineNumbers && this.push(new I("w:suppressLineNumbers", e3.suppressLineNumbers));
          }
          push(e3) {
            this.root.push(e3);
          }
          prepForXml(e3) {
            if (e3.viewWrapper instanceof yn)
              for (const t3 of this.numberingReferences)
                e3.file.Numbering.createConcreteNumberingInstance(t3.reference, t3.instance);
            return super.prepForXml(e3);
          }
        }
        class In extends s {
          constructor(e3) {
            if (super("w:p"), "string" == typeof e3)
              return this.properties = new Sn({}), this.root.push(this.properties), this.root.push(new we(e3)), this;
            if (this.properties = new Sn(e3), this.root.push(this.properties), e3.text && this.root.push(new we(e3.text)), e3.children)
              for (const t3 of e3.children)
                if (t3 instanceof Or) {
                  this.root.push(t3.start);
                  for (const e4 of t3.children)
                    this.root.push(e4);
                  this.root.push(t3.end);
                } else
                  this.root.push(t3);
          }
          prepForXml(e3) {
            for (const t3 of this.root)
              if (t3 instanceof Rr) {
                const r2 = this.root.indexOf(t3), n2 = new Sr(t3.options.children, (0, Ke.EL)());
                e3.viewWrapper.Relationships.createRelationship(n2.linkId, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", t3.options.link, Ce.EXTERNAL), this.root[r2] = n2;
              }
            return super.prepForXml(e3);
          }
          addRunToFront(e3) {
            return this.root.splice(1, 0, e3), this;
          }
        }
        class Rn extends s {
          constructor(e3) {
            super("m:oMath");
            for (const t3 of e3.children)
              this.root.push(t3);
          }
        }
        class Nn extends s {
          constructor(e3) {
            super("m:t"), this.root.push(e3);
          }
        }
        class Cn extends s {
          constructor(e3) {
            super("m:r"), this.root.push(new Nn(e3));
          }
        }
        class On extends s {
          constructor(e3) {
            super("m:den");
            for (const t3 of e3)
              this.root.push(t3);
          }
        }
        class kn extends s {
          constructor(e3) {
            super("m:num");
            for (const t3 of e3)
              this.root.push(t3);
          }
        }
        class Ln extends s {
          constructor(e3) {
            super("m:f"), this.root.push(new kn(e3.numerator)), this.root.push(new On(e3.denominator));
          }
        }
        class Dn extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { accent: "m:val" };
          }
        }
        class Pn extends s {
          constructor(e3) {
            super("m:chr"), this.root.push(new Dn({ accent: e3 }));
          }
        }
        class Fn extends s {
          constructor(e3) {
            super("m:e");
            for (const t3 of e3)
              this.root.push(t3);
          }
        }
        class Bn extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { value: "m:val" };
          }
        }
        class Mn extends s {
          constructor() {
            super("m:limLoc"), this.root.push(new Bn({ value: "undOvr" }));
          }
        }
        class Un extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { hide: "m:val" };
          }
        }
        class Hn extends s {
          constructor() {
            super("m:subHide"), this.root.push(new Un({ hide: 1 }));
          }
        }
        class zn extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { hide: "m:val" };
          }
        }
        class jn extends s {
          constructor() {
            super("m:supHide"), this.root.push(new zn({ hide: 1 }));
          }
        }
        class Wn extends s {
          constructor(e3, t3, r2) {
            super("m:naryPr"), e3 && this.root.push(new Pn(e3)), this.root.push(new Mn()), t3 || this.root.push(new jn()), r2 || this.root.push(new Hn());
          }
        }
        class Kn extends s {
          constructor(e3) {
            super("m:sub");
            for (const t3 of e3)
              this.root.push(t3);
          }
        }
        class Gn extends s {
          constructor(e3) {
            super("m:sup");
            for (const t3 of e3)
              this.root.push(t3);
          }
        }
        class Vn extends s {
          constructor(e3) {
            super("m:nary"), this.root.push(new Wn("\u2211", !!e3.superScript, !!e3.subScript)), e3.subScript && this.root.push(new Kn(e3.subScript)), e3.superScript && this.root.push(new Gn(e3.superScript)), this.root.push(new Fn(e3.children));
          }
        }
        class $n extends s {
          constructor(e3) {
            super("m:nary"), this.root.push(new Wn("", !!e3.superScript, !!e3.subScript)), e3.subScript && this.root.push(new Kn(e3.subScript)), e3.superScript && this.root.push(new Gn(e3.superScript)), this.root.push(new Fn(e3.children));
          }
        }
        class Xn extends s {
          constructor() {
            super("m:sSupPr");
          }
        }
        class qn extends s {
          constructor(e3) {
            super("m:sSup"), this.root.push(new Xn()), this.root.push(new Fn(e3.children)), this.root.push(new Gn(e3.superScript));
          }
        }
        class Zn extends s {
          constructor() {
            super("m:sSubPr");
          }
        }
        class Yn extends s {
          constructor(e3) {
            super("m:sSub"), this.root.push(new Zn()), this.root.push(new Fn(e3.children)), this.root.push(new Kn(e3.subScript));
          }
        }
        class Jn extends s {
          constructor() {
            super("m:sSubSupPr");
          }
        }
        class Qn extends s {
          constructor(e3) {
            super("m:sSubSup"), this.root.push(new Jn()), this.root.push(new Fn(e3.children)), this.root.push(new Kn(e3.subScript)), this.root.push(new Gn(e3.superScript));
          }
        }
        class es extends s {
          constructor() {
            super("m:sPrePr");
          }
        }
        class ts extends s {
          constructor(e3) {
            super("m:sPre"), this.root.push(new es()), this.root.push(new Fn(e3.children)), this.root.push(new Kn(e3.subScript)), this.root.push(new Gn(e3.superScript));
          }
        }
        const rs = "";
        class ns extends s {
          constructor(e3) {
            if (super("m:deg"), e3)
              for (const t3 of e3)
                this.root.push(t3);
          }
        }
        class ss extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { hide: "m:val" };
          }
        }
        class is extends s {
          constructor() {
            super("m:degHide"), this.root.push(new ss({ hide: 1 }));
          }
        }
        class os extends s {
          constructor(e3) {
            super("m:radPr"), e3 || this.root.push(new is());
          }
        }
        class as extends s {
          constructor(e3) {
            super("m:rad"), this.root.push(new os(!!e3.degree)), this.root.push(new ns(e3.degree)), this.root.push(new Fn(e3.children));
          }
        }
        class cs extends s {
          constructor(e3) {
            super("m:fName");
            for (const t3 of e3)
              this.root.push(t3);
          }
        }
        class us extends s {
          constructor() {
            super("m:funcPr");
          }
        }
        class ls extends s {
          constructor(e3) {
            super("m:func"), this.root.push(new us()), this.root.push(new cs(e3.name)), this.root.push(new Fn(e3.children));
          }
        }
        class hs extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { character: "m:val" };
          }
        }
        class ps extends s {
          constructor(e3) {
            super("m:begChr"), this.root.push(new hs({ character: e3 }));
          }
        }
        class ds extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { character: "m:val" };
          }
        }
        class fs extends s {
          constructor(e3) {
            super("m:endChr"), this.root.push(new ds({ character: e3 }));
          }
        }
        class ms extends s {
          constructor(e3) {
            super("m:dPr"), e3 && (this.root.push(new ps(e3.beginningCharacter)), this.root.push(new fs(e3.endingCharacter)));
          }
        }
        class ws extends s {
          constructor(e3) {
            super("m:d"), this.root.push(new ms()), this.root.push(new Fn(e3.children));
          }
        }
        class gs extends s {
          constructor(e3) {
            super("m:d"), this.root.push(new ms({ beginningCharacter: "[", endingCharacter: "]" })), this.root.push(new Fn(e3.children));
          }
        }
        class ys extends s {
          constructor(e3) {
            super("m:d"), this.root.push(new ms({ beginningCharacter: "{", endingCharacter: "}" })), this.root.push(new Fn(e3.children));
          }
        }
        class bs extends s {
          constructor(e3) {
            super("m:d"), this.root.push(new ms({ beginningCharacter: "\u2329", endingCharacter: "\u232A" })), this.root.push(new Fn(e3.children));
          }
        }
        class xs extends s {
          constructor(e3) {
            super("w:tblGrid");
            for (const t3 of e3)
              this.root.push(new _s(t3));
          }
        }
        class vs extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { w: "w:w" };
          }
        }
        class _s extends s {
          constructor(e3) {
            super("w:gridCol"), void 0 !== e3 && this.root.push(new vs({ w: (0, S.Jd)(e3) }));
          }
        }
        var Es, Ts, As, Ss;
        !function(e3) {
          e3.AUTO = "auto", e3.DXA = "dxa", e3.NIL = "nil", e3.PERCENTAGE = "pct";
        }(Es || (Es = {}));
        class Is extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { type: "w:type", size: "w:w" };
          }
        }
        class Rs extends s {
          constructor(e3, { type: t3 = Es.AUTO, size: r2 }) {
            super(e3);
            let n2 = r2;
            t3 === Es.PERCENTAGE && "number" == typeof r2 && (n2 = `${r2}%`), this.root.push(new Is({ type: t3, size: (0, S.aB)(n2) }));
          }
        }
        !function(e3) {
          e3.TABLE = "w:tblCellMar", e3.TABLE_CELL = "w:tcMar";
        }(Ts || (Ts = {}));
        class Ns extends i {
          constructor(e3, { marginUnitType: t3 = Es.DXA, top: r2, left: n2, bottom: s2, right: i2 }) {
            super(e3), void 0 !== r2 && this.root.push(new Rs("w:top", { type: t3, size: r2 })), void 0 !== n2 && this.root.push(new Rs("w:left", { type: t3, size: n2 })), void 0 !== s2 && this.root.push(new Rs("w:bottom", { type: t3, size: s2 })), void 0 !== i2 && this.root.push(new Rs("w:right", { type: t3, size: i2 }));
          }
        }
        class Cs extends i {
          constructor(e3) {
            super("w:tcBorders"), e3.top && this.root.push(new P("w:top", e3.top)), e3.start && this.root.push(new P("w:start", e3.start)), e3.left && this.root.push(new P("w:left", e3.left)), e3.bottom && this.root.push(new P("w:bottom", e3.bottom)), e3.end && this.root.push(new P("w:end", e3.end)), e3.right && this.root.push(new P("w:right", e3.right));
          }
        }
        class Os extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { val: "w:val" };
          }
        }
        class ks extends s {
          constructor(e3) {
            super("w:gridSpan"), this.root.push(new Os({ val: (0, S.vH)(e3) }));
          }
        }
        !function(e3) {
          e3.CONTINUE = "continue", e3.RESTART = "restart";
        }(As || (As = {}));
        class Ls extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { val: "w:val" };
          }
        }
        class Ds extends s {
          constructor(e3) {
            super("w:vMerge"), this.root.push(new Ls({ val: e3 }));
          }
        }
        !function(e3) {
          e3.BOTTOM_TO_TOP_LEFT_TO_RIGHT = "btLr", e3.LEFT_TO_RIGHT_TOP_TO_BOTTOM = "lrTb", e3.TOP_TO_BOTTOM_RIGHT_TO_LEFT = "tbRl";
        }(Ss || (Ss = {}));
        class Ps extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { val: "w:val" };
          }
        }
        class Fs extends s {
          constructor(e3) {
            super("w:textDirection"), this.root.push(new Ps({ val: e3 }));
          }
        }
        class Bs extends i {
          constructor(e3) {
            super("w:tcPr"), e3.width && this.root.push(new Rs("w:tcW", e3.width)), e3.columnSpan && this.root.push(new ks(e3.columnSpan)), e3.verticalMerge ? this.root.push(new Ds(e3.verticalMerge)) : e3.rowSpan && e3.rowSpan > 1 && this.root.push(new Ds(As.RESTART)), e3.borders && this.root.push(new Cs(e3.borders)), e3.shading && this.root.push(new Y(e3.shading)), e3.margins && this.root.push(new Ns(Ts.TABLE_CELL, e3.margins)), e3.textDirection && this.root.push(new Fs(e3.textDirection)), e3.verticalAlign && this.root.push(new Mr(e3.verticalAlign));
          }
        }
        class Ms extends s {
          constructor(e3) {
            super("w:tc"), this.options = e3, this.root.push(new Bs(e3));
            for (const t3 of e3.children)
              this.root.push(t3);
          }
          prepForXml(e3) {
            return this.root[this.root.length - 1] instanceof In || this.root.push(new In({})), super.prepForXml(e3);
          }
        }
        const Us = { style: w.NONE, size: 0, color: "auto" }, Hs = { style: w.SINGLE, size: 4, color: "auto" };
        class zs extends s {
          constructor(e3) {
            super("w:tblBorders"), e3.top ? this.root.push(new P("w:top", e3.top)) : this.root.push(new P("w:top", Hs)), e3.left ? this.root.push(new P("w:left", e3.left)) : this.root.push(new P("w:left", Hs)), e3.bottom ? this.root.push(new P("w:bottom", e3.bottom)) : this.root.push(new P("w:bottom", Hs)), e3.right ? this.root.push(new P("w:right", e3.right)) : this.root.push(new P("w:right", Hs)), e3.insideHorizontal ? this.root.push(new P("w:insideH", e3.insideHorizontal)) : this.root.push(new P("w:insideH", Hs)), e3.insideVertical ? this.root.push(new P("w:insideV", e3.insideVertical)) : this.root.push(new P("w:insideV", Hs));
          }
        }
        zs.NONE = { top: Us, bottom: Us, left: Us, right: Us, insideHorizontal: Us, insideVertical: Us };
        var js, Ws, Ks, Gs, Vs, $s, Xs;
        !function(e3) {
          e3.MARGIN = "margin", e3.PAGE = "page", e3.TEXT = "text";
        }(js || (js = {})), function(e3) {
          e3.CENTER = "center", e3.INSIDE = "inside", e3.LEFT = "left", e3.OUTSIDE = "outside", e3.RIGHT = "right";
        }(Ws || (Ws = {})), function(e3) {
          e3.CENTER = "center", e3.INSIDE = "inside", e3.BOTTOM = "bottom", e3.OUTSIDE = "outside", e3.INLINE = "inline", e3.TOP = "top";
        }(Ks || (Ks = {})), function(e3) {
          e3.NEVER = "never", e3.OVERLAP = "overlap";
        }(Gs || (Gs = {}));
        class qs extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { horizontalAnchor: "w:horzAnchor", verticalAnchor: "w:vertAnchor", absoluteHorizontalPosition: "w:tblpX", relativeHorizontalPosition: "w:tblpXSpec", absoluteVerticalPosition: "w:tblpY", relativeVerticalPosition: "w:tblpYSpec", bottomFromText: "w:bottomFromText", topFromText: "w:topFromText", leftFromText: "w:leftFromText", rightFromText: "w:rightFromText" };
          }
        }
        class Zs extends s {
          constructor(e3) {
            var { leftFromText: t3, rightFromText: r2, topFromText: n2, bottomFromText: s2, absoluteHorizontalPosition: i2, absoluteVerticalPosition: o2 } = e3, a2 = function(e4, t4) {
              var r3 = {};
              for (var n3 in e4)
                Object.prototype.hasOwnProperty.call(e4, n3) && t4.indexOf(n3) < 0 && (r3[n3] = e4[n3]);
              if (null != e4 && "function" == typeof Object.getOwnPropertySymbols) {
                var s3 = 0;
                for (n3 = Object.getOwnPropertySymbols(e4); s3 < n3.length; s3++)
                  t4.indexOf(n3[s3]) < 0 && Object.prototype.propertyIsEnumerable.call(e4, n3[s3]) && (r3[n3[s3]] = e4[n3[s3]]);
              }
              return r3;
            }(e3, ["leftFromText", "rightFromText", "topFromText", "bottomFromText", "absoluteHorizontalPosition", "absoluteVerticalPosition"]);
            super("w:tblpPr"), this.root.push(new qs(Object.assign({ leftFromText: void 0 === t3 ? void 0 : (0, S.Jd)(t3), rightFromText: void 0 === r2 ? void 0 : (0, S.Jd)(r2), topFromText: void 0 === n2 ? void 0 : (0, S.Jd)(n2), bottomFromText: void 0 === s2 ? void 0 : (0, S.Jd)(s2), absoluteHorizontalPosition: void 0 === i2 ? void 0 : (0, S.xb)(i2), absoluteVerticalPosition: void 0 === o2 ? void 0 : (0, S.xb)(o2) }, a2))), a2.overlap && this.root.push(new O("w:tblOverlap", a2.overlap));
          }
        }
        !function(e3) {
          e3.AUTOFIT = "autofit", e3.FIXED = "fixed";
        }(Vs || (Vs = {}));
        class Ys extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { type: "w:type" };
          }
        }
        class Js extends s {
          constructor(e3) {
            super("w:tblLayout"), this.root.push(new Ys({ type: e3 }));
          }
        }
        class Qs extends i {
          constructor(e3) {
            super("w:tblPr"), e3.style && this.root.push(new N("w:tblStyle", e3.style)), e3.float && this.root.push(new Zs(e3.float)), void 0 !== e3.visuallyRightToLeft && this.root.push(new I("w:bidiVisual", e3.visuallyRightToLeft)), e3.width && this.root.push(new Rs("w:tblW", e3.width)), e3.alignment && this.root.push(new D(e3.alignment)), e3.indent && this.root.push(new Rs("w:tblInd", e3.indent)), e3.borders && this.root.push(new zs(e3.borders)), e3.shading && this.root.push(new Y(e3.shading)), e3.layout && this.root.push(new Js(e3.layout)), e3.cellMargin && this.root.push(new Ns(Ts.TABLE, e3.cellMargin));
          }
        }
        class ei extends s {
          constructor({ rows: e3, width: t3, columnWidths: r2 = Array(Math.max(...e3.map((e4) => e4.CellCount))).fill(100), margins: n2, indent: s2, float: i2, layout: o2, style: a2, borders: c2, alignment: u2, visuallyRightToLeft: l2 }) {
            super("w:tbl"), this.root.push(new Qs({ borders: null != c2 ? c2 : {}, width: null != t3 ? t3 : { size: 100 }, indent: s2, float: i2, layout: o2, style: a2, alignment: u2, cellMargin: n2, visuallyRightToLeft: l2 })), this.root.push(new xs(r2));
            for (const t4 of e3)
              this.root.push(t4);
            e3.forEach((t4, r3) => {
              if (r3 === e3.length - 1)
                return;
              let n3 = 0;
              t4.cells.forEach((t5) => {
                if (t5.options.rowSpan && t5.options.rowSpan > 1) {
                  const s3 = new Ms({ rowSpan: t5.options.rowSpan - 1, columnSpan: t5.options.columnSpan, borders: t5.options.borders, children: [], verticalMerge: As.CONTINUE });
                  e3[r3 + 1].addCellToColumnIndex(s3, n3);
                }
                n3 += t5.options.columnSpan || 1;
              });
            });
          }
        }
        !function(e3) {
          e3.AUTO = "auto", e3.ATLEAST = "atLeast", e3.EXACT = "exact";
        }($s || ($s = {}));
        class ti extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { value: "w:val", rule: "w:hRule" };
          }
        }
        class ri extends s {
          constructor(e3, t3) {
            super("w:trHeight"), this.root.push(new ti({ value: (0, S.Jd)(e3), rule: t3 }));
          }
        }
        class ni extends i {
          constructor(e3) {
            super("w:trPr"), void 0 !== e3.cantSplit && this.root.push(new I("w:cantSplit", e3.cantSplit)), void 0 !== e3.tableHeader && this.root.push(new I("w:tblHeader", e3.tableHeader)), e3.height && this.root.push(new ri(e3.height.value, e3.height.rule));
          }
        }
        class si extends s {
          constructor(e3) {
            super("w:tr"), this.options = e3, this.root.push(new ni(e3));
            for (const t3 of e3.children)
              this.root.push(t3);
          }
          get CellCount() {
            return this.options.children.length;
          }
          get cells() {
            return this.root.filter((e3) => e3 instanceof Ms);
          }
          addCellToIndex(e3, t3) {
            this.root.splice(t3 + 1, 0, e3);
          }
          addCellToColumnIndex(e3, t3) {
            const r2 = this.columnIndexToRootIndex(t3, true);
            this.addCellToIndex(e3, r2 - 1);
          }
          rootIndexToColumnIndex(e3) {
            if (e3 < 1 || e3 >= this.root.length)
              throw new Error("cell 'rootIndex' should between 1 to " + (this.root.length - 1));
            let t3 = 0;
            for (let r2 = 1; r2 < e3; r2++)
              t3 += this.root[r2].options.columnSpan || 1;
            return t3;
          }
          columnIndexToRootIndex(e3, t3 = false) {
            if (e3 < 0)
              throw new Error("cell 'columnIndex' should not less than zero");
            let r2 = 0, n2 = 1;
            for (; r2 <= e3; ) {
              if (n2 >= this.root.length) {
                if (t3)
                  return this.root.length;
                throw new Error("cell 'columnIndex' should not great than " + (r2 - 1));
              }
              const e4 = this.root[n2];
              n2 += 1, r2 += e4 && e4.options.columnSpan || 1;
            }
            return n2 - 1;
          }
        }
        class ii extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { xmlns: "xmlns", vt: "xmlns:vt" };
          }
        }
        class oi extends s {
          constructor() {
            super("Properties"), this.root.push(new ii({ xmlns: "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties", vt: "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes" }));
          }
        }
        class ai extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { xmlns: "xmlns" };
          }
        }
        class ci extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { contentType: "ContentType", extension: "Extension" };
          }
        }
        class ui extends s {
          constructor(e3, t3) {
            super("Default"), this.root.push(new ci({ contentType: e3, extension: t3 }));
          }
        }
        class li extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { contentType: "ContentType", partName: "PartName" };
          }
        }
        class hi extends s {
          constructor(e3, t3) {
            super("Override"), this.root.push(new li({ contentType: e3, partName: t3 }));
          }
        }
        class pi extends s {
          constructor() {
            super("Types"), this.root.push(new ai({ xmlns: "http://schemas.openxmlformats.org/package/2006/content-types" })), this.root.push(new ui("image/png", "png")), this.root.push(new ui("image/jpeg", "jpeg")), this.root.push(new ui("image/jpeg", "jpg")), this.root.push(new ui("image/bmp", "bmp")), this.root.push(new ui("image/gif", "gif")), this.root.push(new ui("application/vnd.openxmlformats-package.relationships+xml", "rels")), this.root.push(new ui("application/xml", "xml")), this.root.push(new hi("application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml", "/word/document.xml")), this.root.push(new hi("application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml", "/word/styles.xml")), this.root.push(new hi("application/vnd.openxmlformats-package.core-properties+xml", "/docProps/core.xml")), this.root.push(new hi("application/vnd.openxmlformats-officedocument.custom-properties+xml", "/docProps/custom.xml")), this.root.push(new hi("application/vnd.openxmlformats-officedocument.extended-properties+xml", "/docProps/app.xml")), this.root.push(new hi("application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml", "/word/numbering.xml")), this.root.push(new hi("application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml", "/word/footnotes.xml")), this.root.push(new hi("application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml", "/word/settings.xml")), this.root.push(new hi("application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml", "/word/comments.xml"));
          }
          addFooter(e3) {
            this.root.push(new hi("application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml", `/word/footer${e3}.xml`));
          }
          addHeader(e3) {
            this.root.push(new hi("application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml", `/word/header${e3}.xml`));
          }
        }
        class di extends s {
          constructor(e3) {
            super("cp:coreProperties"), this.root.push(new pn({ cp: "http://schemas.openxmlformats.org/package/2006/metadata/core-properties", dc: "http://purl.org/dc/elements/1.1/", dcterms: "http://purl.org/dc/terms/", dcmitype: "http://purl.org/dc/dcmitype/", xsi: "http://www.w3.org/2001/XMLSchema-instance" })), e3.title && this.root.push(new k("dc:title", e3.title)), e3.subject && this.root.push(new k("dc:subject", e3.subject)), e3.creator && this.root.push(new k("dc:creator", e3.creator)), e3.keywords && this.root.push(new k("cp:keywords", e3.keywords)), e3.description && this.root.push(new k("dc:description", e3.description)), e3.lastModifiedBy && this.root.push(new k("cp:lastModifiedBy", e3.lastModifiedBy)), e3.revision && this.root.push(new k("cp:revision", String(e3.revision))), this.root.push(new fi("dcterms:created")), this.root.push(new fi("dcterms:modified"));
          }
        }
        class fi extends s {
          constructor(e3) {
            super(e3), this.root.push(new pn({ type: "dcterms:W3CDTF" })), this.root.push((0, S.sF)(/* @__PURE__ */ new Date()));
          }
        }
        class mi extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { xmlns: "xmlns", vt: "xmlns:vt" };
          }
        }
        class wi extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { fmtid: "fmtid", pid: "pid", name: "name" };
          }
        }
        class gi extends s {
          constructor(e3, t3) {
            super("property"), this.root.push(new wi({ fmtid: "{D5CDD505-2E9C-101B-9397-08002B2CF9AE}", pid: e3.toString(), name: t3.name })), this.root.push(new yi(t3.value));
          }
        }
        class yi extends s {
          constructor(e3) {
            super("vt:lpwstr"), this.root.push(e3);
          }
        }
        class bi extends s {
          constructor(e3) {
            super("Properties"), this.properties = [], this.root.push(new mi({ xmlns: "http://schemas.openxmlformats.org/officeDocument/2006/custom-properties", vt: "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes" })), this.nextId = 2;
            for (const t3 of e3)
              this.addCustomProperty(t3);
          }
          prepForXml(e3) {
            return this.properties.forEach((e4) => this.root.push(e4)), super.prepForXml(e3);
          }
          addCustomProperty(e3) {
            this.properties.push(new gi(this.nextId++, e3));
          }
        }
        class xi extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { wpc: "xmlns:wpc", mc: "xmlns:mc", o: "xmlns:o", r: "xmlns:r", m: "xmlns:m", v: "xmlns:v", wp14: "xmlns:wp14", wp: "xmlns:wp", w10: "xmlns:w10", w: "xmlns:w", w14: "xmlns:w14", w15: "xmlns:w15", wpg: "xmlns:wpg", wpi: "xmlns:wpi", wne: "xmlns:wne", wps: "xmlns:wps", cp: "xmlns:cp", dc: "xmlns:dc", dcterms: "xmlns:dcterms", dcmitype: "xmlns:dcmitype", xsi: "xmlns:xsi", type: "xsi:type" };
          }
        }
        class vi extends f {
          constructor(e3, t3) {
            super("w:ftr", t3), this.refId = e3, t3 || this.root.push(new xi({ wpc: "http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas", mc: "http://schemas.openxmlformats.org/markup-compatibility/2006", o: "urn:schemas-microsoft-com:office:office", r: "http://schemas.openxmlformats.org/officeDocument/2006/relationships", m: "http://schemas.openxmlformats.org/officeDocument/2006/math", v: "urn:schemas-microsoft-com:vml", wp14: "http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing", wp: "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing", w10: "urn:schemas-microsoft-com:office:word", w: "http://schemas.openxmlformats.org/wordprocessingml/2006/main", w14: "http://schemas.microsoft.com/office/word/2010/wordml", w15: "http://schemas.microsoft.com/office/word/2012/wordml", wpg: "http://schemas.microsoft.com/office/word/2010/wordprocessingGroup", wpi: "http://schemas.microsoft.com/office/word/2010/wordprocessingInk", wne: "http://schemas.microsoft.com/office/word/2006/wordml", wps: "http://schemas.microsoft.com/office/word/2010/wordprocessingShape" }));
          }
          get ReferenceId() {
            return this.refId;
          }
          add(e3) {
            this.root.push(e3);
          }
        }
        class _i {
          constructor(e3, t3, r2) {
            this.media = e3, this.footer = new vi(t3, r2), this.relationships = new gn();
          }
          add(e3) {
            this.footer.add(e3);
          }
          addChildElement(e3) {
            this.footer.addChildElement(e3);
          }
          get View() {
            return this.footer;
          }
          get Relationships() {
            return this.relationships;
          }
          get Media() {
            return this.media;
          }
        }
        class Ei extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { type: "w:type", id: "w:id" };
          }
        }
        class Ti extends s {
          constructor() {
            super("w:footnoteRef");
          }
        }
        class Ai extends me {
          constructor() {
            super({ style: "FootnoteReference" }), this.root.push(new Ti());
          }
        }
        !function(e3) {
          e3.SEPERATOR = "separator", e3.CONTINUATION_SEPERATOR = "continuationSeparator";
        }(Xs || (Xs = {}));
        class Si extends s {
          constructor(e3) {
            super("w:footnote"), this.root.push(new Ei({ type: e3.type, id: e3.id }));
            for (let t3 = 0; t3 < e3.children.length; t3++) {
              const r2 = e3.children[t3];
              0 === t3 && r2.addRunToFront(new Ai()), this.root.push(r2);
            }
          }
        }
        class Ii extends s {
          constructor() {
            super("w:continuationSeparator");
          }
        }
        class Ri extends me {
          constructor() {
            super({}), this.root.push(new Ii());
          }
        }
        class Ni extends s {
          constructor() {
            super("w:separator");
          }
        }
        class Ci extends me {
          constructor() {
            super({}), this.root.push(new Ni());
          }
        }
        class Oi extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { wpc: "xmlns:wpc", mc: "xmlns:mc", o: "xmlns:o", r: "xmlns:r", m: "xmlns:m", v: "xmlns:v", wp14: "xmlns:wp14", wp: "xmlns:wp", w10: "xmlns:w10", w: "xmlns:w", w14: "xmlns:w14", w15: "xmlns:w15", wpg: "xmlns:wpg", wpi: "xmlns:wpi", wne: "xmlns:wne", wps: "xmlns:wps", Ignorable: "mc:Ignorable" };
          }
        }
        class ki extends s {
          constructor() {
            super("w:footnotes"), this.root.push(new Oi({ wpc: "http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas", mc: "http://schemas.openxmlformats.org/markup-compatibility/2006", o: "urn:schemas-microsoft-com:office:office", r: "http://schemas.openxmlformats.org/officeDocument/2006/relationships", m: "http://schemas.openxmlformats.org/officeDocument/2006/math", v: "urn:schemas-microsoft-com:vml", wp14: "http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing", wp: "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing", w10: "urn:schemas-microsoft-com:office:word", w: "http://schemas.openxmlformats.org/wordprocessingml/2006/main", w14: "http://schemas.microsoft.com/office/word/2010/wordml", w15: "http://schemas.microsoft.com/office/word/2012/wordml", wpg: "http://schemas.microsoft.com/office/word/2010/wordprocessingGroup", wpi: "http://schemas.microsoft.com/office/word/2010/wordprocessingInk", wne: "http://schemas.microsoft.com/office/word/2006/wordml", wps: "http://schemas.microsoft.com/office/word/2010/wordprocessingShape", Ignorable: "w14 w15 wp14" }));
            const e3 = new Si({ id: -1, type: Xs.SEPERATOR, children: [new In({ spacing: { after: 0, line: 240, lineRule: Ae.AUTO }, children: [new Ci()] })] });
            this.root.push(e3);
            const t3 = new Si({ id: 0, type: Xs.CONTINUATION_SEPERATOR, children: [new In({ spacing: { after: 0, line: 240, lineRule: Ae.AUTO }, children: [new Ri()] })] });
            this.root.push(t3);
          }
          createFootNote(e3, t3) {
            const r2 = new Si({ id: e3, children: t3 });
            this.root.push(r2);
          }
        }
        class Li {
          constructor() {
            this.footnotess = new ki(), this.relationships = new gn();
          }
          get View() {
            return this.footnotess;
          }
          get Relationships() {
            return this.relationships;
          }
        }
        class Di extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { wpc: "xmlns:wpc", mc: "xmlns:mc", o: "xmlns:o", r: "xmlns:r", m: "xmlns:m", v: "xmlns:v", wp14: "xmlns:wp14", wp: "xmlns:wp", w10: "xmlns:w10", w: "xmlns:w", w14: "xmlns:w14", w15: "xmlns:w15", wpg: "xmlns:wpg", wpi: "xmlns:wpi", wne: "xmlns:wne", wps: "xmlns:wps", cp: "xmlns:cp", dc: "xmlns:dc", dcterms: "xmlns:dcterms", dcmitype: "xmlns:dcmitype", xsi: "xmlns:xsi", type: "xsi:type", cx: "xmlns:cx", cx1: "xmlns:cx1", cx2: "xmlns:cx2", cx3: "xmlns:cx3", cx4: "xmlns:cx4", cx5: "xmlns:cx5", cx6: "xmlns:cx6", cx7: "xmlns:cx7", cx8: "xmlns:cx8", w16cid: "xmlns:w16cid", w16se: "xmlns:w16se" };
          }
        }
        class Pi extends f {
          constructor(e3, t3) {
            super("w:hdr", t3), this.refId = e3, t3 || this.root.push(new Di({ wpc: "http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas", mc: "http://schemas.openxmlformats.org/markup-compatibility/2006", o: "urn:schemas-microsoft-com:office:office", r: "http://schemas.openxmlformats.org/officeDocument/2006/relationships", m: "http://schemas.openxmlformats.org/officeDocument/2006/math", v: "urn:schemas-microsoft-com:vml", wp14: "http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing", wp: "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing", w10: "urn:schemas-microsoft-com:office:word", w: "http://schemas.openxmlformats.org/wordprocessingml/2006/main", w14: "http://schemas.microsoft.com/office/word/2010/wordml", w15: "http://schemas.microsoft.com/office/word/2012/wordml", wpg: "http://schemas.microsoft.com/office/word/2010/wordprocessingGroup", wpi: "http://schemas.microsoft.com/office/word/2010/wordprocessingInk", wne: "http://schemas.microsoft.com/office/word/2006/wordml", wps: "http://schemas.microsoft.com/office/word/2010/wordprocessingShape", cx: "http://schemas.microsoft.com/office/drawing/2014/chartex", cx1: "http://schemas.microsoft.com/office/drawing/2015/9/8/chartex", cx2: "http://schemas.microsoft.com/office/drawing/2015/10/21/chartex", cx3: "http://schemas.microsoft.com/office/drawing/2016/5/9/chartex", cx4: "http://schemas.microsoft.com/office/drawing/2016/5/10/chartex", cx5: "http://schemas.microsoft.com/office/drawing/2016/5/11/chartex", cx6: "http://schemas.microsoft.com/office/drawing/2016/5/12/chartex", cx7: "http://schemas.microsoft.com/office/drawing/2016/5/13/chartex", cx8: "http://schemas.microsoft.com/office/drawing/2016/5/14/chartex", w16cid: "http://schemas.microsoft.com/office/word/2016/wordml/cid", w16se: "http://schemas.microsoft.com/office/word/2015/wordml/symex" }));
          }
          get ReferenceId() {
            return this.refId;
          }
          add(e3) {
            this.root.push(e3);
          }
        }
        class Fi {
          constructor(e3, t3, r2) {
            this.media = e3, this.header = new Pi(t3, r2), this.relationships = new gn();
          }
          add(e3) {
            return this.header.add(e3), this;
          }
          addChildElement(e3) {
            this.header.addChildElement(e3);
          }
          get View() {
            return this.header;
          }
          get Relationships() {
            return this.relationships;
          }
          get Media() {
            return this.media;
          }
        }
        class Bi {
          constructor() {
            this.map = /* @__PURE__ */ new Map();
          }
          addMedia(e3, t3) {
            const r2 = `${(0, Ke.EL)()}.png`, n2 = { stream: "string" == typeof e3 ? this.convertDataURIToBinary(e3) : e3, fileName: r2, transformation: { pixels: { x: Math.round(t3.width), y: Math.round(t3.height) }, emus: { x: Math.round(9525 * t3.width), y: Math.round(9525 * t3.height) }, flip: t3.flip, rotation: t3.rotation ? 6e4 * t3.rotation : void 0 } };
            return this.map.set(r2, n2), n2;
          }
          addImage(e3, t3) {
            this.map.set(e3, t3);
          }
          get Array() {
            return Array.from(this.map.values());
          }
          convertDataURIToBinary(e3) {
            const t3 = ";base64,", n2 = e3.indexOf(t3) + t3.length;
            return "function" == typeof atob ? new Uint8Array(atob(e3.substring(n2)).split("").map((e4) => e4.charCodeAt(0))) : new (r(8764)).Buffer(e3, "base64");
          }
        }
        const Mi = "";
        var Ui, Hi;
        !function(e3) {
          e3.DECIMAL = "decimal", e3.UPPER_ROMAN = "upperRoman", e3.LOWER_ROMAN = "lowerRoman", e3.UPPER_LETTER = "upperLetter", e3.LOWER_LETTER = "lowerLetter", e3.ORDINAL = "ordinal", e3.CARDINAL_TEXT = "cardinalText", e3.ORDINAL_TEXT = "ordinalText", e3.HEX = "hex", e3.CHICAGO = "chicago", e3.IDEOGRAPH__DIGITAL = "ideographDigital", e3.JAPANESE_COUNTING = "japaneseCounting", e3.AIUEO = "aiueo", e3.IROHA = "iroha", e3.DECIMAL_FULL_WIDTH = "decimalFullWidth", e3.DECIMAL_HALF_WIDTH = "decimalHalfWidth", e3.JAPANESE_LEGAL = "japaneseLegal", e3.JAPANESE_DIGITAL_TEN_THOUSAND = "japaneseDigitalTenThousand", e3.DECIMAL_ENCLOSED_CIRCLE = "decimalEnclosedCircle", e3.DECIMAL_FULL_WIDTH2 = "decimalFullWidth2", e3.AIUEO_FULL_WIDTH = "aiueoFullWidth", e3.IROHA_FULL_WIDTH = "irohaFullWidth", e3.DECIMAL_ZERO = "decimalZero", e3.BULLET = "bullet", e3.GANADA = "ganada", e3.CHOSUNG = "chosung", e3.DECIMAL_ENCLOSED_FULLSTOP = "decimalEnclosedFullstop", e3.DECIMAL_ENCLOSED_PARENTHESES = "decimalEnclosedParen", e3.DECIMAL_ENCLOSED_CIRCLE_CHINESE = "decimalEnclosedCircleChinese", e3.IDEOGRAPH_ENCLOSED_CIRCLE = "ideographEnclosedCircle", e3.IDEOGRAPH_TRADITIONAL = "ideographTraditional", e3.IDEOGRAPH_ZODIAC = "ideographZodiac", e3.IDEOGRAPH_ZODIAC_TRADITIONAL = "ideographZodiacTraditional", e3.TAIWANESE_COUNTING = "taiwaneseCounting", e3.IDEOGRAPH_LEGAL_TRADITIONAL = "ideographLegalTraditional", e3.TAIWANESE_COUNTING_THOUSAND = "taiwaneseCountingThousand", e3.TAIWANESE_DIGITAL = "taiwaneseDigital", e3.CHINESE_COUNTING = "chineseCounting", e3.CHINESE_LEGAL_SIMPLIFIED = "chineseLegalSimplified", e3.CHINESE_COUNTING_THOUSAND = "chineseCountingThousand", e3.KOREAN_DIGITAL = "koreanDigital", e3.KOREAN_COUNTING = "koreanCounting", e3.KOREAN_LEGAL = "koreanLegal", e3.KOREAN_DIGITAL2 = "koreanDigital2", e3.VIETNAMESE_COUNTING = "vietnameseCounting", e3.RUSSIAN_LOWER = "russianLower", e3.RUSSIAN_UPPER = "russianUpper", e3.NONE = "none", e3.NUMBER_IN_DASH = "numberInDash", e3.HEBREW1 = "hebrew1", e3.HEBREW2 = "hebrew2", e3.ARABIC_ALPHA = "arabicAlpha", e3.ARABIC_ABJAD = "arabicAbjad", e3.HINDI_VOWELS = "hindiVowels", e3.HINDI_CONSONANTS = "hindiConsonants", e3.HINDI_NUMBERS = "hindiNumbers", e3.HINDI_COUNTING = "hindiCounting", e3.THAI_LETTERS = "thaiLetters", e3.THAI_NUMBERS = "thaiNumbers", e3.THAI_COUNTING = "thaiCounting", e3.BAHT_TEXT = "bahtText", e3.DOLLAR_TEXT = "dollarText", e3.CUSTOM = "custom";
        }(Ui || (Ui = {}));
        class zi extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { ilvl: "w:ilvl", tentative: "w15:tentative" };
          }
        }
        class ji extends s {
          constructor(e3) {
            super("w:numFmt"), this.root.push(new a({ val: e3 }));
          }
        }
        class Wi extends s {
          constructor(e3) {
            super("w:lvlText"), this.root.push(new a({ val: e3 }));
          }
        }
        class Ki extends s {
          constructor(e3) {
            super("w:lvlJc"), this.root.push(new a({ val: e3 }));
          }
        }
        !function(e3) {
          e3.NOTHING = "nothing", e3.SPACE = "space", e3.TAB = "tab";
        }(Hi || (Hi = {}));
        class Gi extends s {
          constructor(e3) {
            super("w:suff"), this.root.push(new a({ val: e3 }));
          }
        }
        class Vi extends s {
          constructor() {
            super("w:isLgl");
          }
        }
        class $i extends s {
          constructor({ level: e3, format: t3, text: r2, alignment: n2 = m.START, start: s2 = 1, style: i2, suffix: o2, isLegalNumberingStyle: a2 }) {
            if (super("w:lvl"), this.root.push(new C("w:start", (0, S.vH)(s2))), t3 && this.root.push(new ji(t3)), o2 && this.root.push(new Gi(o2)), a2 && this.root.push(new Vi()), r2 && this.root.push(new Wi(r2)), this.root.push(new Ki(n2)), this.paragraphProperties = new Sn(i2 && i2.paragraph), this.runProperties = new pe(i2 && i2.run), this.root.push(this.paragraphProperties), this.root.push(this.runProperties), e3 > 9)
              throw new Error("Level cannot be greater than 9. Read more here: https://answers.microsoft.com/en-us/msoffice/forum/all/does-word-support-more-than-9-list-levels/d130fdcd-1781-446d-8c84-c6c79124e4d7");
            this.root.push(new zi({ ilvl: (0, S.vH)(e3), tentative: 1 }));
          }
        }
        class Xi extends $i {
        }
        class qi extends $i {
        }
        class Zi extends s {
          constructor(e3) {
            super("w:multiLevelType"), this.root.push(new a({ val: e3 }));
          }
        }
        class Yi extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { abstractNumId: "w:abstractNumId", restartNumberingAfterBreak: "w15:restartNumberingAfterBreak" };
          }
        }
        class Ji extends s {
          constructor(e3, t3) {
            super("w:abstractNum"), this.root.push(new Yi({ abstractNumId: (0, S.vH)(e3), restartNumberingAfterBreak: 0 })), this.root.push(new Zi("hybridMultilevel")), this.id = e3;
            for (const e4 of t3)
              this.root.push(new Xi(e4));
          }
        }
        class Qi extends s {
          constructor(e3) {
            super("w:abstractNumId"), this.root.push(new a({ val: e3 }));
          }
        }
        class eo extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { numId: "w:numId" };
          }
        }
        class to extends s {
          constructor(e3) {
            super("w:num"), this.numId = e3.numId, this.reference = e3.reference, this.instance = e3.instance, this.root.push(new eo({ numId: (0, S.vH)(e3.numId) })), this.root.push(new Qi((0, S.vH)(e3.abstractNumId))), e3.overrideLevel && this.root.push(new no(e3.overrideLevel.num, e3.overrideLevel.start));
          }
        }
        class ro extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { ilvl: "w:ilvl" };
          }
        }
        class no extends s {
          constructor(e3, t3) {
            super("w:lvlOverride"), this.root.push(new ro({ ilvl: e3 })), void 0 !== t3 && this.root.push(new io(t3));
          }
        }
        class so extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { val: "w:val" };
          }
        }
        class io extends s {
          constructor(e3) {
            super("w:startOverride"), this.root.push(new so({ val: e3 }));
          }
        }
        class oo extends s {
          constructor(e3) {
            super("w:numbering"), this.abstractNumberingMap = /* @__PURE__ */ new Map(), this.concreteNumberingMap = /* @__PURE__ */ new Map(), this.referenceConfigMap = /* @__PURE__ */ new Map(), this.root.push(new pn({ wpc: "http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas", mc: "http://schemas.openxmlformats.org/markup-compatibility/2006", o: "urn:schemas-microsoft-com:office:office", r: "http://schemas.openxmlformats.org/officeDocument/2006/relationships", m: "http://schemas.openxmlformats.org/officeDocument/2006/math", v: "urn:schemas-microsoft-com:vml", wp14: "http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing", wp: "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing", w10: "urn:schemas-microsoft-com:office:word", w: "http://schemas.openxmlformats.org/wordprocessingml/2006/main", w14: "http://schemas.microsoft.com/office/word/2010/wordml", w15: "http://schemas.microsoft.com/office/word/2012/wordml", wpg: "http://schemas.microsoft.com/office/word/2010/wordprocessingGroup", wpi: "http://schemas.microsoft.com/office/word/2010/wordprocessingInk", wne: "http://schemas.microsoft.com/office/word/2006/wordml", wps: "http://schemas.microsoft.com/office/word/2010/wordprocessingShape", Ignorable: "w14 w15 wp14" }));
            const t3 = new Ji((0, Ke.NY)(), [{ level: 0, format: Ui.BULLET, text: "\u25CF", alignment: m.LEFT, style: { paragraph: { indent: { left: (0, Ke.vw)(0.5), hanging: (0, Ke.vw)(0.25) } } } }, { level: 1, format: Ui.BULLET, text: "\u25CB", alignment: m.LEFT, style: { paragraph: { indent: { left: (0, Ke.vw)(1), hanging: (0, Ke.vw)(0.25) } } } }, { level: 2, format: Ui.BULLET, text: "\u25A0", alignment: m.LEFT, style: { paragraph: { indent: { left: 2160, hanging: (0, Ke.vw)(0.25) } } } }, { level: 3, format: Ui.BULLET, text: "\u25CF", alignment: m.LEFT, style: { paragraph: { indent: { left: 2880, hanging: (0, Ke.vw)(0.25) } } } }, { level: 4, format: Ui.BULLET, text: "\u25CB", alignment: m.LEFT, style: { paragraph: { indent: { left: 3600, hanging: (0, Ke.vw)(0.25) } } } }, { level: 5, format: Ui.BULLET, text: "\u25A0", alignment: m.LEFT, style: { paragraph: { indent: { left: 4320, hanging: (0, Ke.vw)(0.25) } } } }, { level: 6, format: Ui.BULLET, text: "\u25CF", alignment: m.LEFT, style: { paragraph: { indent: { left: 5040, hanging: (0, Ke.vw)(0.25) } } } }, { level: 7, format: Ui.BULLET, text: "\u25CF", alignment: m.LEFT, style: { paragraph: { indent: { left: 5760, hanging: (0, Ke.vw)(0.25) } } } }, { level: 8, format: Ui.BULLET, text: "\u25CF", alignment: m.LEFT, style: { paragraph: { indent: { left: 6480, hanging: (0, Ke.vw)(0.25) } } } }]);
            this.concreteNumberingMap.set("default-bullet-numbering", new to({ numId: 1, abstractNumId: t3.id, reference: "default-bullet-numbering", instance: 0, overrideLevel: { num: 0, start: 1 } })), this.abstractNumberingMap.set("default-bullet-numbering", t3);
            for (const t4 of e3.config)
              this.abstractNumberingMap.set(t4.reference, new Ji((0, Ke.NY)(), t4.levels)), this.referenceConfigMap.set(t4.reference, t4.levels);
          }
          prepForXml(e3) {
            for (const e4 of this.abstractNumberingMap.values())
              this.root.push(e4);
            for (const e4 of this.concreteNumberingMap.values())
              this.root.push(e4);
            return super.prepForXml(e3);
          }
          createConcreteNumberingInstance(e3, t3) {
            const r2 = this.abstractNumberingMap.get(e3);
            if (!r2)
              return;
            const n2 = `${e3}-${t3}`;
            if (this.concreteNumberingMap.has(n2))
              return;
            const s2 = this.referenceConfigMap.get(e3), i2 = s2 && s2[0].start, o2 = { numId: (0, Ke.NY)(), abstractNumId: r2.id, reference: e3, instance: t3, overrideLevel: i2 && Number.isInteger(i2) ? { num: 0, start: i2 } : { num: 0, start: 1 } };
            this.concreteNumberingMap.set(n2, new to(o2));
          }
          get ConcreteNumbering() {
            return Array.from(this.concreteNumberingMap.values());
          }
          get ReferenceConfig() {
            return Array.from(this.referenceConfigMap.values());
          }
        }
        class ao extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { version: "w:val", name: "w:name", uri: "w:uri" };
          }
        }
        class co extends s {
          constructor(e3) {
            super("w:compatSetting"), this.root.push(new ao({ version: e3, uri: "http://schemas.microsoft.com/office/word", name: "compatibilityMode" }));
          }
        }
        class uo extends s {
          constructor(e3) {
            super("w:compat"), e3.version && this.root.push(new co(e3.version)), e3.useSingleBorderforContiguousCells && this.root.push(new I("w:useSingleBorderforContiguousCells", e3.useSingleBorderforContiguousCells)), e3.wordPerfectJustification && this.root.push(new I("w:wpJustification", e3.wordPerfectJustification)), e3.noTabStopForHangingIndent && this.root.push(new I("w:noTabHangInd", e3.noTabStopForHangingIndent)), e3.noLeading && this.root.push(new I("w:noLeading", e3.noLeading)), e3.spaceForUnderline && this.root.push(new I("w:spaceForUL", e3.spaceForUnderline)), e3.noColumnBalance && this.root.push(new I("w:noColumnBalance", e3.noColumnBalance)), e3.balanceSingleByteDoubleByteWidth && this.root.push(new I("w:balanceSingleByteDoubleByteWidth", e3.balanceSingleByteDoubleByteWidth)), e3.noExtraLineSpacing && this.root.push(new I("w:noExtraLineSpacing", e3.noExtraLineSpacing)), e3.doNotLeaveBackslashAlone && this.root.push(new I("w:doNotLeaveBackslashAlone", e3.doNotLeaveBackslashAlone)), e3.underlineTrailingSpaces && this.root.push(new I("w:ulTrailSpace", e3.underlineTrailingSpaces)), e3.doNotExpandShiftReturn && this.root.push(new I("w:doNotExpandShiftReturn", e3.doNotExpandShiftReturn)), e3.spacingInWholePoints && this.root.push(new I("w:spacingInWholePoints", e3.spacingInWholePoints)), e3.lineWrapLikeWord6 && this.root.push(new I("w:lineWrapLikeWord6", e3.lineWrapLikeWord6)), e3.printBodyTextBeforeHeader && this.root.push(new I("w:printBodyTextBeforeHeader", e3.printBodyTextBeforeHeader)), e3.printColorsBlack && this.root.push(new I("w:printColBlack", e3.printColorsBlack)), e3.spaceWidth && this.root.push(new I("w:wpSpaceWidth", e3.spaceWidth)), e3.showBreaksInFrames && this.root.push(new I("w:showBreaksInFrames", e3.showBreaksInFrames)), e3.subFontBySize && this.root.push(new I("w:subFontBySize", e3.subFontBySize)), e3.suppressBottomSpacing && this.root.push(new I("w:suppressBottomSpacing", e3.suppressBottomSpacing)), e3.suppressTopSpacing && this.root.push(new I("w:suppressTopSpacing", e3.suppressTopSpacing)), e3.suppressSpacingAtTopOfPage && this.root.push(new I("w:suppressSpacingAtTopOfPage", e3.suppressSpacingAtTopOfPage)), e3.suppressTopSpacingWP && this.root.push(new I("w:suppressTopSpacingWP", e3.suppressTopSpacingWP)), e3.suppressSpBfAfterPgBrk && this.root.push(new I("w:suppressSpBfAfterPgBrk", e3.suppressSpBfAfterPgBrk)), e3.swapBordersFacingPages && this.root.push(new I("w:swapBordersFacingPages", e3.swapBordersFacingPages)), e3.convertMailMergeEsc && this.root.push(new I("w:convMailMergeEsc", e3.convertMailMergeEsc)), e3.truncateFontHeightsLikeWP6 && this.root.push(new I("w:truncateFontHeightsLikeWP6", e3.truncateFontHeightsLikeWP6)), e3.macWordSmallCaps && this.root.push(new I("w:mwSmallCaps", e3.macWordSmallCaps)), e3.usePrinterMetrics && this.root.push(new I("w:usePrinterMetrics", e3.usePrinterMetrics)), e3.doNotSuppressParagraphBorders && this.root.push(new I("w:doNotSuppressParagraphBorders", e3.doNotSuppressParagraphBorders)), e3.wrapTrailSpaces && this.root.push(new I("w:wrapTrailSpaces", e3.wrapTrailSpaces)), e3.footnoteLayoutLikeWW8 && this.root.push(new I("w:footnoteLayoutLikeWW8", e3.footnoteLayoutLikeWW8)), e3.shapeLayoutLikeWW8 && this.root.push(new I("w:shapeLayoutLikeWW8", e3.shapeLayoutLikeWW8)), e3.alignTablesRowByRow && this.root.push(new I("w:alignTablesRowByRow", e3.alignTablesRowByRow)), e3.forgetLastTabAlignment && this.root.push(new I("w:forgetLastTabAlignment", e3.forgetLastTabAlignment)), e3.adjustLineHeightInTable && this.root.push(new I("w:adjustLineHeightInTable", e3.adjustLineHeightInTable)), e3.autoSpaceLikeWord95 && this.root.push(new I("w:autoSpaceLikeWord95", e3.autoSpaceLikeWord95)), e3.noSpaceRaiseLower && this.root.push(new I("w:noSpaceRaiseLower", e3.noSpaceRaiseLower)), e3.doNotUseHTMLParagraphAutoSpacing && this.root.push(new I("w:doNotUseHTMLParagraphAutoSpacing", e3.doNotUseHTMLParagraphAutoSpacing)), e3.layoutRawTableWidth && this.root.push(new I("w:layoutRawTableWidth", e3.layoutRawTableWidth)), e3.layoutTableRowsApart && this.root.push(new I("w:layoutTableRowsApart", e3.layoutTableRowsApart)), e3.useWord97LineBreakRules && this.root.push(new I("w:useWord97LineBreakRules", e3.useWord97LineBreakRules)), e3.doNotBreakWrappedTables && this.root.push(new I("w:doNotBreakWrappedTables", e3.doNotBreakWrappedTables)), e3.doNotSnapToGridInCell && this.root.push(new I("w:doNotSnapToGridInCell", e3.doNotSnapToGridInCell)), e3.selectFieldWithFirstOrLastCharacter && this.root.push(new I("w:selectFldWithFirstOrLastChar", e3.selectFieldWithFirstOrLastCharacter)), e3.applyBreakingRules && this.root.push(new I("w:applyBreakingRules", e3.applyBreakingRules)), e3.doNotWrapTextWithPunctuation && this.root.push(new I("w:doNotWrapTextWithPunct", e3.doNotWrapTextWithPunctuation)), e3.doNotUseEastAsianBreakRules && this.root.push(new I("w:doNotUseEastAsianBreakRules", e3.doNotUseEastAsianBreakRules)), e3.useWord2002TableStyleRules && this.root.push(new I("w:useWord2002TableStyleRules", e3.useWord2002TableStyleRules)), e3.growAutofit && this.root.push(new I("w:growAutofit", e3.growAutofit)), e3.useFELayout && this.root.push(new I("w:useFELayout", e3.useFELayout)), e3.useNormalStyleForList && this.root.push(new I("w:useNormalStyleForList", e3.useNormalStyleForList)), e3.doNotUseIndentAsNumberingTabStop && this.root.push(new I("w:doNotUseIndentAsNumberingTabStop", e3.doNotUseIndentAsNumberingTabStop)), e3.useAlternateEastAsianLineBreakRules && this.root.push(new I("w:useAltKinsokuLineBreakRules", e3.useAlternateEastAsianLineBreakRules)), e3.allowSpaceOfSameStyleInTable && this.root.push(new I("w:allowSpaceOfSameStyleInTable", e3.allowSpaceOfSameStyleInTable)), e3.doNotSuppressIndentation && this.root.push(new I("w:doNotSuppressIndentation", e3.doNotSuppressIndentation)), e3.doNotAutofitConstrainedTables && this.root.push(new I("w:doNotAutofitConstrainedTables", e3.doNotAutofitConstrainedTables)), e3.autofitToFirstFixedWidthCell && this.root.push(new I("w:autofitToFirstFixedWidthCell", e3.autofitToFirstFixedWidthCell)), e3.underlineTabInNumberingList && this.root.push(new I("w:underlineTabInNumList", e3.underlineTabInNumberingList)), e3.displayHangulFixedWidth && this.root.push(new I("w:displayHangulFixedWidth", e3.displayHangulFixedWidth)), e3.splitPgBreakAndParaMark && this.root.push(new I("w:splitPgBreakAndParaMark", e3.splitPgBreakAndParaMark)), e3.doNotVerticallyAlignCellWithSp && this.root.push(new I("w:doNotVertAlignCellWithSp", e3.doNotVerticallyAlignCellWithSp)), e3.doNotBreakConstrainedForcedTable && this.root.push(new I("w:doNotBreakConstrainedForcedTable", e3.doNotBreakConstrainedForcedTable)), e3.ignoreVerticalAlignmentInTextboxes && this.root.push(new I("w:doNotVertAlignInTxbx", e3.ignoreVerticalAlignmentInTextboxes)), e3.useAnsiKerningPairs && this.root.push(new I("w:useAnsiKerningPairs", e3.useAnsiKerningPairs)), e3.cachedColumnBalance && this.root.push(new I("w:cachedColBalance", e3.cachedColumnBalance));
          }
        }
        class lo extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { wpc: "xmlns:wpc", mc: "xmlns:mc", o: "xmlns:o", r: "xmlns:r", m: "xmlns:m", v: "xmlns:v", wp14: "xmlns:wp14", wp: "xmlns:wp", w10: "xmlns:w10", w: "xmlns:w", w14: "xmlns:w14", w15: "xmlns:w15", wpg: "xmlns:wpg", wpi: "xmlns:wpi", wne: "xmlns:wne", wps: "xmlns:wps", Ignorable: "mc:Ignorable" };
          }
        }
        class ho extends s {
          constructor(e3) {
            var t3, r2, n2, s2;
            super("w:settings"), this.root.push(new lo({ wpc: "http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas", mc: "http://schemas.openxmlformats.org/markup-compatibility/2006", o: "urn:schemas-microsoft-com:office:office", r: "http://schemas.openxmlformats.org/officeDocument/2006/relationships", m: "http://schemas.openxmlformats.org/officeDocument/2006/math", v: "urn:schemas-microsoft-com:vml", wp14: "http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing", wp: "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing", w10: "urn:schemas-microsoft-com:office:word", w: "http://schemas.openxmlformats.org/wordprocessingml/2006/main", w14: "http://schemas.microsoft.com/office/word/2010/wordml", w15: "http://schemas.microsoft.com/office/word/2012/wordml", wpg: "http://schemas.microsoft.com/office/word/2010/wordprocessingGroup", wpi: "http://schemas.microsoft.com/office/word/2010/wordprocessingInk", wne: "http://schemas.microsoft.com/office/word/2006/wordml", wps: "http://schemas.microsoft.com/office/word/2010/wordprocessingShape", Ignorable: "w14 w15 wp14" })), this.root.push(new I("w:displayBackgroundShape", true)), void 0 !== e3.trackRevisions && this.root.push(new I("w:trackRevisions", e3.trackRevisions)), void 0 !== e3.evenAndOddHeaders && this.root.push(new I("w:evenAndOddHeaders", e3.evenAndOddHeaders)), void 0 !== e3.updateFields && this.root.push(new I("w:updateFields", e3.updateFields)), this.root.push(new uo(Object.assign(Object.assign({}, null !== (t3 = e3.compatibility) && void 0 !== t3 ? t3 : {}), { version: null !== (s2 = null !== (n2 = null === (r2 = e3.compatibility) || void 0 === r2 ? void 0 : r2.version) && void 0 !== n2 ? n2 : e3.compatibilityModeVersion) && void 0 !== s2 ? s2 : 15 })));
          }
        }
        class po extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { val: "w:val" };
          }
        }
        class fo extends s {
          constructor(e3) {
            super("w:name"), this.root.push(new po({ val: e3 }));
          }
        }
        class mo extends s {
          constructor(e3) {
            super("w:uiPriority"), this.root.push(new po({ val: (0, S.vH)(e3) }));
          }
        }
        class wo extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { type: "w:type", styleId: "w:styleId", default: "w:default", customStyle: "w:customStyle" };
          }
        }
        class go extends s {
          constructor(e3, t3) {
            super("w:style"), this.root.push(new wo(e3)), t3.name && this.root.push(new fo(t3.name)), t3.basedOn && this.root.push(new N("w:basedOn", t3.basedOn)), t3.next && this.root.push(new N("w:next", t3.next)), t3.link && this.root.push(new N("w:link", t3.link)), void 0 !== t3.uiPriority && this.root.push(new mo(t3.uiPriority)), void 0 !== t3.semiHidden && this.root.push(new I("w:semiHidden", t3.semiHidden)), void 0 !== t3.unhideWhenUsed && this.root.push(new I("w:unhideWhenUsed", t3.unhideWhenUsed)), void 0 !== t3.quickFormat && this.root.push(new I("w:qFormat", t3.quickFormat));
          }
        }
        class yo extends go {
          constructor(e3) {
            super({ type: "paragraph", styleId: e3.id }, e3), this.paragraphProperties = new Sn(e3.paragraph), this.runProperties = new pe(e3.run), this.root.push(this.paragraphProperties), this.root.push(this.runProperties);
          }
        }
        class bo extends go {
          constructor(e3) {
            super({ type: "character", styleId: e3.id }, Object.assign({ uiPriority: 99, unhideWhenUsed: true }, e3)), this.runProperties = new pe(e3.run), this.root.push(this.runProperties);
          }
        }
        class xo extends yo {
          constructor(e3) {
            super(Object.assign(Object.assign({}, e3), { basedOn: "Normal", next: "Normal", quickFormat: true }));
          }
        }
        class vo extends xo {
          constructor(e3) {
            super(Object.assign(Object.assign({}, e3), { id: "Title", name: "Title" }));
          }
        }
        class _o extends xo {
          constructor(e3) {
            super(Object.assign(Object.assign({}, e3), { id: "Heading1", name: "Heading 1" }));
          }
        }
        class Eo extends xo {
          constructor(e3) {
            super(Object.assign(Object.assign({}, e3), { id: "Heading2", name: "Heading 2" }));
          }
        }
        class To extends xo {
          constructor(e3) {
            super(Object.assign(Object.assign({}, e3), { id: "Heading3", name: "Heading 3" }));
          }
        }
        class Ao extends xo {
          constructor(e3) {
            super(Object.assign(Object.assign({}, e3), { id: "Heading4", name: "Heading 4" }));
          }
        }
        class So extends xo {
          constructor(e3) {
            super(Object.assign(Object.assign({}, e3), { id: "Heading5", name: "Heading 5" }));
          }
        }
        class Io extends xo {
          constructor(e3) {
            super(Object.assign(Object.assign({}, e3), { id: "Heading6", name: "Heading 6" }));
          }
        }
        class Ro extends xo {
          constructor(e3) {
            super(Object.assign(Object.assign({}, e3), { id: "Strong", name: "Strong" }));
          }
        }
        class No extends yo {
          constructor(e3) {
            super(Object.assign(Object.assign({}, e3), { id: "ListParagraph", name: "List Paragraph", basedOn: "Normal", quickFormat: true }));
          }
        }
        class Co extends yo {
          constructor(e3) {
            super(Object.assign(Object.assign({}, e3), { id: "FootnoteText", name: "footnote text", link: "FootnoteTextChar", basedOn: "Normal", uiPriority: 99, semiHidden: true, unhideWhenUsed: true, paragraph: { spacing: { after: 0, line: 240, lineRule: Ae.AUTO } }, run: { size: 20 } }));
          }
        }
        class Oo extends bo {
          constructor(e3) {
            super(Object.assign(Object.assign({}, e3), { id: "FootnoteReference", name: "footnote reference", basedOn: "DefaultParagraphFont", semiHidden: true, run: { superScript: true } }));
          }
        }
        class ko extends bo {
          constructor(e3) {
            super(Object.assign(Object.assign({}, e3), { id: "FootnoteTextChar", name: "Footnote Text Char", basedOn: "DefaultParagraphFont", link: "FootnoteText", semiHidden: true, run: { size: 20 } }));
          }
        }
        class Lo extends bo {
          constructor(e3) {
            super(Object.assign(Object.assign({}, e3), { id: "Hyperlink", name: "Hyperlink", basedOn: "DefaultParagraphFont", run: { color: "0563C1", underline: { type: T.SINGLE } } }));
          }
        }
        class Do extends s {
          constructor(e3) {
            if (super("w:styles"), e3.initialStyles && this.root.push(e3.initialStyles), e3.importedStyles)
              for (const t3 of e3.importedStyles)
                this.root.push(t3);
            if (e3.paragraphStyles)
              for (const t3 of e3.paragraphStyles)
                this.root.push(new yo(t3));
            if (e3.characterStyles)
              for (const t3 of e3.characterStyles)
                this.root.push(new bo(t3));
          }
        }
        class Po extends s {
          constructor(e3) {
            super("w:pPrDefault"), this.root.push(new Sn(e3));
          }
        }
        class Fo extends s {
          constructor(e3) {
            super("w:rPrDefault"), this.root.push(new pe(e3));
          }
        }
        class Bo extends s {
          constructor(e3) {
            super("w:docDefaults"), this.runPropertiesDefaults = new Fo(e3.run), this.paragraphPropertiesDefaults = new Po(e3.paragraph), this.root.push(this.runPropertiesDefaults), this.root.push(this.paragraphPropertiesDefaults);
          }
        }
        class Mo {
          newInstance(e3) {
            const t3 = (0, c.xml2js)(e3, { compact: false });
            let r2;
            for (const e4 of t3.elements || [])
              "w:styles" === e4.name && (r2 = e4);
            if (void 0 === r2)
              throw new Error("can not find styles element");
            const n2 = r2.elements || [];
            return new Do({ initialStyles: new p(r2.attributes), importedStyles: n2.map((e4) => u(e4)) });
          }
        }
        class Uo {
          newInstance(e3 = {}) {
            var t3;
            return { initialStyles: new pn({ mc: "http://schemas.openxmlformats.org/markup-compatibility/2006", r: "http://schemas.openxmlformats.org/officeDocument/2006/relationships", w: "http://schemas.openxmlformats.org/wordprocessingml/2006/main", w14: "http://schemas.microsoft.com/office/word/2010/wordml", w15: "http://schemas.microsoft.com/office/word/2012/wordml", Ignorable: "w14 w15" }), importedStyles: [new Bo(null !== (t3 = e3.document) && void 0 !== t3 ? t3 : {}), new vo(Object.assign({ run: { size: 56 } }, e3.title)), new _o(Object.assign({ run: { color: "2E74B5", size: 32 } }, e3.heading1)), new Eo(Object.assign({ run: { color: "2E74B5", size: 26 } }, e3.heading2)), new To(Object.assign({ run: { color: "1F4D78", size: 24 } }, e3.heading3)), new Ao(Object.assign({ run: { color: "2E74B5", italics: true } }, e3.heading4)), new So(Object.assign({ run: { color: "2E74B5" } }, e3.heading5)), new Io(Object.assign({ run: { color: "1F4D78" } }, e3.heading6)), new Ro(Object.assign({ run: { bold: true } }, e3.strong)), new No(e3.listParagraph || {}), new Lo(e3.hyperlink || {}), new Oo(e3.footnoteReference || {}), new Co(e3.footnoteText || {}), new ko(e3.footnoteTextChar || {})] };
          }
        }
        class Ho {
          constructor(e3, t3 = {}) {
            var r2, n2, s2, i2, o2, a2, c2;
            if (this.currentRelationshipId = 1, this.headers = [], this.footers = [], this.coreProperties = new di(Object.assign(Object.assign({}, e3), { creator: null !== (r2 = e3.creator) && void 0 !== r2 ? r2 : "Un-named", revision: null !== (n2 = e3.revision) && void 0 !== n2 ? n2 : 1, lastModifiedBy: null !== (s2 = e3.lastModifiedBy) && void 0 !== s2 ? s2 : "Un-named" })), this.numbering = new oo(e3.numbering ? e3.numbering : { config: [] }), this.comments = new ur(null !== (i2 = e3.comments) && void 0 !== i2 ? i2 : { children: [] }), this.fileRelationships = new gn(), this.customProperties = new bi(null !== (o2 = e3.customProperties) && void 0 !== o2 ? o2 : []), this.appProperties = new oi(), this.footnotesWrapper = new Li(), this.contentTypes = new pi(), this.documentWrapper = new yn({ background: e3.background }), this.settings = new ho({ compatibilityModeVersion: e3.compatabilityModeVersion, compatibility: e3.compatibility, evenAndOddHeaders: !!e3.evenAndOddHeaderAndFooters, trackRevisions: null === (a2 = e3.features) || void 0 === a2 ? void 0 : a2.trackRevisions, updateFields: null === (c2 = e3.features) || void 0 === c2 ? void 0 : c2.updateFields }), this.media = t3.template && t3.template.media ? t3.template.media : new Bi(), t3.template && (this.currentRelationshipId = t3.template.currentRelationshipId + 1), t3.template && e3.externalStyles)
              throw Error("can not use both template and external styles");
            if (t3.template && t3.template.styles) {
              const e4 = new Mo();
              this.styles = e4.newInstance(t3.template.styles);
            } else if (e3.externalStyles) {
              const t4 = new Mo();
              this.styles = t4.newInstance(e3.externalStyles);
            } else if (e3.styles) {
              const t4 = new Uo().newInstance(e3.styles.default);
              this.styles = new Do(Object.assign(Object.assign({}, t4), e3.styles));
            } else {
              const e4 = new Uo();
              this.styles = new Do(e4.newInstance());
            }
            if (this.addDefaultRelationships(), t3.template && t3.template.headers)
              for (const e4 of t3.template.headers)
                this.addHeaderToDocument(e4.header, e4.type);
            if (t3.template && t3.template.footers)
              for (const e4 of t3.template.footers)
                this.addFooterToDocument(e4.footer, e4.type);
            for (const t4 of e3.sections)
              this.addSection(t4);
            if (e3.footnotes)
              for (const t4 in e3.footnotes)
                this.footnotesWrapper.View.createFootNote(parseFloat(t4), e3.footnotes[t4].children);
          }
          addSection({ headers: e3 = {}, footers: t3 = {}, children: r2, properties: n2 }) {
            this.documentWrapper.View.Body.addSection(Object.assign(Object.assign({}, n2), { headerWrapperGroup: { default: e3.default ? this.createHeader(e3.default) : void 0, first: e3.first ? this.createHeader(e3.first) : void 0, even: e3.even ? this.createHeader(e3.even) : void 0 }, footerWrapperGroup: { default: t3.default ? this.createFooter(t3.default) : void 0, first: t3.first ? this.createFooter(t3.first) : void 0, even: t3.even ? this.createFooter(t3.even) : void 0 } }));
            for (const e4 of r2)
              this.documentWrapper.View.add(e4);
          }
          createHeader(e3) {
            const t3 = new Fi(this.media, this.currentRelationshipId++);
            for (const r2 of e3.options.children)
              t3.add(r2);
            return this.addHeaderToDocument(t3), t3;
          }
          createFooter(e3) {
            const t3 = new _i(this.media, this.currentRelationshipId++);
            for (const r2 of e3.options.children)
              t3.add(r2);
            return this.addFooterToDocument(t3), t3;
          }
          addHeaderToDocument(e3, t3 = Le.DEFAULT) {
            this.headers.push({ header: e3, type: t3 }), this.documentWrapper.Relationships.createRelationship(e3.View.ReferenceId, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/header", `header${this.headers.length}.xml`), this.contentTypes.addHeader(this.headers.length);
          }
          addFooterToDocument(e3, t3 = Le.DEFAULT) {
            this.footers.push({ footer: e3, type: t3 }), this.documentWrapper.Relationships.createRelationship(e3.View.ReferenceId, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer", `footer${this.footers.length}.xml`), this.contentTypes.addFooter(this.footers.length);
          }
          addDefaultRelationships() {
            this.fileRelationships.createRelationship(1, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument", "word/document.xml"), this.fileRelationships.createRelationship(2, "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties", "docProps/core.xml"), this.fileRelationships.createRelationship(3, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties", "docProps/app.xml"), this.fileRelationships.createRelationship(4, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/custom-properties", "docProps/custom.xml"), this.documentWrapper.Relationships.createRelationship(this.currentRelationshipId++, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles", "styles.xml"), this.documentWrapper.Relationships.createRelationship(this.currentRelationshipId++, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering", "numbering.xml"), this.documentWrapper.Relationships.createRelationship(this.currentRelationshipId++, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/footnotes", "footnotes.xml"), this.documentWrapper.Relationships.createRelationship(this.currentRelationshipId++, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings", "settings.xml"), this.documentWrapper.Relationships.createRelationship(this.currentRelationshipId++, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments", "comments.xml");
          }
          get Document() {
            return this.documentWrapper;
          }
          get Styles() {
            return this.styles;
          }
          get CoreProperties() {
            return this.coreProperties;
          }
          get Numbering() {
            return this.numbering;
          }
          get Media() {
            return this.media;
          }
          get FileRelationships() {
            return this.fileRelationships;
          }
          get Headers() {
            return this.headers.map((e3) => e3.header);
          }
          get Footers() {
            return this.footers.map((e3) => e3.footer);
          }
          get ContentTypes() {
            return this.contentTypes;
          }
          get CustomProperties() {
            return this.customProperties;
          }
          get AppProperties() {
            return this.appProperties;
          }
          get FootNotes() {
            return this.footnotesWrapper;
          }
          get Settings() {
            return this.settings;
          }
          get Comments() {
            return this.comments;
          }
        }
        const zo = "";
        class jo extends s {
          constructor(e3 = {}) {
            super("w:instrText"), this.properties = e3, this.root.push(new V({ space: v.PRESERVE }));
            let t3 = "TOC";
            this.properties.captionLabel && (t3 = `${t3} \\a "${this.properties.captionLabel}"`), this.properties.entriesFromBookmark && (t3 = `${t3} \\b "${this.properties.entriesFromBookmark}"`), this.properties.captionLabelIncludingNumbers && (t3 = `${t3} \\c "${this.properties.captionLabelIncludingNumbers}"`), this.properties.sequenceAndPageNumbersSeparator && (t3 = `${t3} \\d "${this.properties.sequenceAndPageNumbersSeparator}"`), this.properties.tcFieldIdentifier && (t3 = `${t3} \\f "${this.properties.tcFieldIdentifier}"`), this.properties.hyperlink && (t3 = `${t3} \\h`), this.properties.tcFieldLevelRange && (t3 = `${t3} \\l "${this.properties.tcFieldLevelRange}"`), this.properties.pageNumbersEntryLevelsRange && (t3 = `${t3} \\n "${this.properties.pageNumbersEntryLevelsRange}"`), this.properties.headingStyleRange && (t3 = `${t3} \\o "${this.properties.headingStyleRange}"`), this.properties.entryAndPageNumberSeparator && (t3 = `${t3} \\p "${this.properties.entryAndPageNumberSeparator}"`), this.properties.seqFieldIdentifierForPrefix && (t3 = `${t3} \\s "${this.properties.seqFieldIdentifierForPrefix}"`), this.properties.stylesWithLevels && this.properties.stylesWithLevels.length && (t3 = `${t3} \\t "${this.properties.stylesWithLevels.map((e4) => `${e4.styleName},${e4.level}`).join(",")}"`), this.properties.useAppliedParagraphOutlineLevel && (t3 = `${t3} \\u`), this.properties.preserveTabInEntries && (t3 = `${t3} \\w`), this.properties.preserveNewLineInEntries && (t3 = `${t3} \\x`), this.properties.hideTabAndPageNumbersInWebView && (t3 = `${t3} \\z`), this.root.push(t3);
          }
        }
        class Wo extends s {
          constructor() {
            super("w:sdtContent");
          }
        }
        class Ko extends s {
          constructor(e3) {
            super("w:sdtPr"), this.root.push(new N("w:alias", e3));
          }
        }
        class Go extends s {
          constructor(e3 = "Table of Contents", t3) {
            super("w:sdt"), this.root.push(new Ko(e3));
            const r2 = new Wo(), n2 = new In({ children: [new me({ children: [new W(true), new jo(t3), new K()] })] });
            r2.addChildElement(n2);
            const s2 = new In({ children: [new me({ children: [new G()] })] });
            r2.addChildElement(s2), this.root.push(r2);
          }
        }
        class Vo {
          constructor(e3, t3) {
            this.styleName = e3, this.level = t3;
          }
        }
        class $o {
          constructor(e3 = { children: [] }) {
            this.options = e3;
          }
        }
        class Xo {
          constructor(e3 = { children: [] }) {
            this.options = e3;
          }
        }
        class qo extends o {
          constructor() {
            super(...arguments), this.xmlKeys = { id: "w:id" };
          }
        }
        class Zo extends s {
          constructor(e3) {
            super("w:footnoteReference"), this.root.push(new qo({ id: e3 }));
          }
        }
        class Yo extends me {
          constructor(e3) {
            super({ style: "FootnoteReference" }), this.root.push(new Zo(e3));
          }
        }
        class Jo extends s {
          constructor(e3) {
            super("w:ins"), this.root.push(new J({ id: e3.id, author: e3.author, date: e3.date })), this.addChildElement(new we(e3));
          }
        }
        class Qo extends s {
          constructor() {
            super("w:delInstrText"), this.root.push(new V({ space: v.PRESERVE })), this.root.push("PAGE");
          }
        }
        class ea extends s {
          constructor() {
            super("w:delInstrText"), this.root.push(new V({ space: v.PRESERVE })), this.root.push("NUMPAGES");
          }
        }
        class ta extends s {
          constructor() {
            super("w:delInstrText"), this.root.push(new V({ space: v.PRESERVE })), this.root.push("SECTIONPAGES");
          }
        }
        class ra extends s {
          constructor(e3) {
            super("w:delText"), this.root.push(new V({ space: v.PRESERVE })), this.root.push(e3);
          }
        }
        class na extends s {
          constructor(e3) {
            super("w:del"), this.root.push(new J({ id: e3.id, author: e3.author, date: e3.date })), this.deletedTextRunWrapper = new sa(e3), this.addChildElement(this.deletedTextRunWrapper);
          }
        }
        class sa extends s {
          constructor(e3) {
            if (super("w:r"), this.root.push(new pe(e3)), e3.children)
              for (const t3 of e3.children)
                if ("string" != typeof t3)
                  this.root.push(t3);
                else
                  switch (t3) {
                    case A.CURRENT:
                      this.root.push(new W()), this.root.push(new Qo()), this.root.push(new K()), this.root.push(new G());
                      break;
                    case A.TOTAL_PAGES:
                      this.root.push(new W()), this.root.push(new ea()), this.root.push(new K()), this.root.push(new G());
                      break;
                    case A.TOTAL_PAGES_IN_SECTION:
                      this.root.push(new W()), this.root.push(new ta()), this.root.push(new K()), this.root.push(new G());
                      break;
                    default:
                      this.root.push(new ra(t3));
                  }
            else
              e3.text && this.root.push(new ra(e3.text));
            if (e3.break)
              for (let t3 = 0; t3 < e3.break; t3++)
                this.root.splice(1, 0, new z());
          }
        }
        var ia = r(6085), oa = r(3479);
        class aa {
          format(e3, t3 = {}) {
            const r2 = e3.prepForXml(t3);
            if (r2)
              return r2;
            throw Error("XMLComponent did not format correctly");
          }
        }
        class ca {
          replace(e3, t3, r2) {
            let n2 = e3;
            return t3.forEach((e4, t4) => {
              n2 = n2.replace(new RegExp(`{${e4.fileName}}`, "g"), (r2 + t4).toString());
            }), n2;
          }
          getMediaData(e3, t3) {
            return t3.Array.filter((t4) => e3.search(`{${t4.fileName}}`) > 0);
          }
        }
        class ua {
          replace(e3, t3) {
            let r2 = e3;
            for (const e4 of t3)
              r2 = r2.replace(new RegExp(`{${e4.reference}-${e4.instance}}`, "g"), e4.numId.toString());
            return r2;
          }
        }
        var la, ha = function(e3, t3, r2, n2) {
          return new (r2 || (r2 = Promise))(function(s2, i2) {
            function o2(e4) {
              try {
                c2(n2.next(e4));
              } catch (e5) {
                i2(e5);
              }
            }
            function a2(e4) {
              try {
                c2(n2.throw(e4));
              } catch (e5) {
                i2(e5);
              }
            }
            function c2(e4) {
              var t4;
              e4.done ? s2(e4.value) : (t4 = e4.value, t4 instanceof r2 ? t4 : new r2(function(e5) {
                e5(t4);
              })).then(o2, a2);
            }
            c2((n2 = n2.apply(e3, t3 || [])).next());
          });
        };
        !function(e3) {
          e3.NONE = "", e3.WITH_2_BLANKS = "  ", e3.WITH_4_BLANKS = "    ", e3.WITH_TAB = "	";
        }(la || (la = {}));
        class pa {
          static toString(e3, t3) {
            return ha(this, void 0, void 0, function* () {
              const r2 = this.compiler.compile(e3, t3);
              return yield r2.generateAsync({ type: "string", mimeType: "application/vnd.openxmlformats-officedocument.wordprocessingml.document", compression: "DEFLATE" });
            });
          }
          static toBuffer(e3, t3) {
            return ha(this, void 0, void 0, function* () {
              const r2 = this.compiler.compile(e3, t3);
              return yield r2.generateAsync({ type: "nodebuffer", mimeType: "application/vnd.openxmlformats-officedocument.wordprocessingml.document", compression: "DEFLATE" });
            });
          }
          static toBase64String(e3, t3) {
            return ha(this, void 0, void 0, function* () {
              const r2 = this.compiler.compile(e3, t3);
              return yield r2.generateAsync({ type: "base64", mimeType: "application/vnd.openxmlformats-officedocument.wordprocessingml.document", compression: "DEFLATE" });
            });
          }
          static toBlob(e3, t3) {
            return ha(this, void 0, void 0, function* () {
              const r2 = this.compiler.compile(e3, t3);
              return yield r2.generateAsync({ type: "blob", mimeType: "application/vnd.openxmlformats-officedocument.wordprocessingml.document", compression: "DEFLATE" });
            });
          }
          static toStream(e3, t3) {
            return this.compiler.compile(e3, t3).generateNodeStream({ type: "nodebuffer", streamFiles: true, mimeType: "application/vnd.openxmlformats-officedocument.wordprocessingml.document", compression: "DEFLATE" });
          }
        }
        pa.compiler = new class {
          constructor() {
            this.formatter = new aa(), this.imageReplacer = new ca(), this.numberingReplacer = new ua();
          }
          compile(e3, t3) {
            const r2 = new ia(), n2 = this.xmlifyFile(e3, t3), s2 = new Map(Object.entries(n2));
            for (const [, e4] of s2)
              if (Array.isArray(e4))
                for (const t4 of e4)
                  r2.file(t4.path, t4.data);
              else
                r2.file(e4.path, e4.data);
            for (const t4 of e3.Media.Array) {
              const e4 = t4.stream;
              r2.file(`word/media/${t4.fileName}`, e4);
            }
            return r2;
          }
          xmlifyFile(e3, t3) {
            const r2 = e3.Document.Relationships.RelationshipCount + 1, n2 = oa(this.formatter.format(e3.Document.View, { viewWrapper: e3.Document, file: e3 }), { indent: t3, declaration: { standalone: "yes", encoding: "UTF-8" } }), s2 = this.imageReplacer.getMediaData(n2, e3.Media);
            return { Relationships: { data: (() => (s2.forEach((t4, n3) => {
              e3.Document.Relationships.createRelationship(r2 + n3, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image", `media/${t4.fileName}`);
            }), oa(this.formatter.format(e3.Document.Relationships, { viewWrapper: e3.Document, file: e3 }), { indent: t3, declaration: { encoding: "UTF-8" } })))(), path: "word/_rels/document.xml.rels" }, Document: { data: (() => {
              const t4 = this.imageReplacer.replace(n2, s2, r2);
              return this.numberingReplacer.replace(t4, e3.Numbering.ConcreteNumbering);
            })(), path: "word/document.xml" }, Styles: { data: (() => {
              const r3 = oa(this.formatter.format(e3.Styles, { viewWrapper: e3.Document, file: e3 }), { indent: t3, declaration: { standalone: "yes", encoding: "UTF-8" } });
              return this.numberingReplacer.replace(r3, e3.Numbering.ConcreteNumbering);
            })(), path: "word/styles.xml" }, Properties: { data: oa(this.formatter.format(e3.CoreProperties, { viewWrapper: e3.Document, file: e3 }), { indent: t3, declaration: { standalone: "yes", encoding: "UTF-8" } }), path: "docProps/core.xml" }, Numbering: { data: oa(this.formatter.format(e3.Numbering, { viewWrapper: e3.Document, file: e3 }), { indent: t3, declaration: { standalone: "yes", encoding: "UTF-8" } }), path: "word/numbering.xml" }, FileRelationships: { data: oa(this.formatter.format(e3.FileRelationships, { viewWrapper: e3.Document, file: e3 }), { indent: t3, declaration: { encoding: "UTF-8" } }), path: "_rels/.rels" }, HeaderRelationships: e3.Headers.map((r3, n3) => {
              const s3 = oa(this.formatter.format(r3.View, { viewWrapper: r3, file: e3 }), { indent: t3, declaration: { encoding: "UTF-8" } });
              return this.imageReplacer.getMediaData(s3, e3.Media).forEach((e4, t4) => {
                r3.Relationships.createRelationship(t4, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image", `media/${e4.fileName}`);
              }), { data: oa(this.formatter.format(r3.Relationships, { viewWrapper: r3, file: e3 }), { indent: t3, declaration: { encoding: "UTF-8" } }), path: `word/_rels/header${n3 + 1}.xml.rels` };
            }), FooterRelationships: e3.Footers.map((r3, n3) => {
              const s3 = oa(this.formatter.format(r3.View, { viewWrapper: r3, file: e3 }), { indent: t3, declaration: { encoding: "UTF-8" } });
              return this.imageReplacer.getMediaData(s3, e3.Media).forEach((e4, t4) => {
                r3.Relationships.createRelationship(t4, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image", `media/${e4.fileName}`);
              }), { data: oa(this.formatter.format(r3.Relationships, { viewWrapper: r3, file: e3 }), { indent: t3, declaration: { encoding: "UTF-8" } }), path: `word/_rels/footer${n3 + 1}.xml.rels` };
            }), Headers: e3.Headers.map((r3, n3) => {
              const s3 = oa(this.formatter.format(r3.View, { viewWrapper: r3, file: e3 }), { indent: t3, declaration: { encoding: "UTF-8" } }), i2 = this.imageReplacer.getMediaData(s3, e3.Media), o2 = this.imageReplacer.replace(s3, i2, 0);
              return { data: this.numberingReplacer.replace(o2, e3.Numbering.ConcreteNumbering), path: `word/header${n3 + 1}.xml` };
            }), Footers: e3.Footers.map((r3, n3) => {
              const s3 = oa(this.formatter.format(r3.View, { viewWrapper: r3, file: e3 }), { indent: t3, declaration: { encoding: "UTF-8" } }), i2 = this.imageReplacer.getMediaData(s3, e3.Media), o2 = this.imageReplacer.replace(s3, i2, 0);
              return { data: this.numberingReplacer.replace(o2, e3.Numbering.ConcreteNumbering), path: `word/footer${n3 + 1}.xml` };
            }), ContentTypes: { data: oa(this.formatter.format(e3.ContentTypes, { viewWrapper: e3.Document, file: e3 }), { indent: t3, declaration: { encoding: "UTF-8" } }), path: "[Content_Types].xml" }, CustomProperties: { data: oa(this.formatter.format(e3.CustomProperties, { viewWrapper: e3.Document, file: e3 }), { indent: t3, declaration: { standalone: "yes", encoding: "UTF-8" } }), path: "docProps/custom.xml" }, AppProperties: { data: oa(this.formatter.format(e3.AppProperties, { viewWrapper: e3.Document, file: e3 }), { indent: t3, declaration: { standalone: "yes", encoding: "UTF-8" } }), path: "docProps/app.xml" }, FootNotes: { data: oa(this.formatter.format(e3.FootNotes.View, { viewWrapper: e3.FootNotes, file: e3 }), { indent: t3, declaration: { encoding: "UTF-8" } }), path: "word/footnotes.xml" }, FootNotesRelationships: { data: oa(this.formatter.format(e3.FootNotes.Relationships, { viewWrapper: e3.FootNotes, file: e3 }), { indent: t3, declaration: { encoding: "UTF-8" } }), path: "word/_rels/footnotes.xml.rels" }, Settings: { data: oa(this.formatter.format(e3.Settings, { viewWrapper: e3.Document, file: e3 }), { indent: t3, declaration: { standalone: "yes", encoding: "UTF-8" } }), path: "word/settings.xml" }, Comments: { data: oa(this.formatter.format(e3.Comments, { viewWrapper: e3.Document, file: e3 }), { indent: t3, declaration: { standalone: "yes", encoding: "UTF-8" } }), path: "word/comments.xml" } };
          }
        }();
        var da = function(e3, t3, r2, n2) {
          return new (r2 || (r2 = Promise))(function(s2, i2) {
            function o2(e4) {
              try {
                c2(n2.next(e4));
              } catch (e5) {
                i2(e5);
              }
            }
            function a2(e4) {
              try {
                c2(n2.throw(e4));
              } catch (e5) {
                i2(e5);
              }
            }
            function c2(e4) {
              var t4;
              e4.done ? s2(e4.value) : (t4 = e4.value, t4 instanceof r2 ? t4 : new r2(function(e5) {
                e5(t4);
              })).then(o2, a2);
            }
            c2((n2 = n2.apply(e3, t3 || [])).next());
          });
        };
        const fa = { "http://schemas.openxmlformats.org/officeDocument/2006/relationships/header": "header", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer": "footer", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image": "image", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink": "hyperlink" };
        var ma;
        !function(e3) {
          e3.HEADER = "header", e3.FOOTER = "footer", e3.IMAGE = "image", e3.HYPERLINK = "hyperlink";
        }(ma || (ma = {}));
        class wa {
          extract(e3) {
            return da(this, void 0, void 0, function* () {
              const t3 = yield ia.loadAsync(e3), r2 = yield t3.files["word/document.xml"].async("text"), n2 = yield t3.files["word/_rels/document.xml.rels"].async("text"), s2 = this.extractDocumentRefs(r2), i2 = this.findReferenceFiles(n2), o2 = new Bi();
              return { headers: yield this.createHeaders(t3, s2, i2, o2, 0), footers: yield this.createFooters(t3, s2, i2, o2, s2.headers.length), currentRelationshipId: s2.footers.length + s2.headers.length, styles: yield t3.files["word/styles.xml"].async("text"), titlePageIsDefined: this.checkIfTitlePageIsDefined(r2), media: o2 };
            });
          }
          createFooters(e3, t3, r2, n2, s2) {
            return da(this, void 0, void 0, function* () {
              const i2 = t3.footers.map((t4, i3) => da(this, void 0, void 0, function* () {
                const o2 = r2.find((e4) => e4.id === t4.id);
                if (null === o2 || !o2)
                  throw new Error(`Can not find target file for id ${t4.id}`);
                const a2 = yield e3.files[`word/${o2.target}`].async("text"), l2 = (0, c.xml2js)(a2, { compact: false, captureSpacesBetweenElements: true });
                if (!l2.elements)
                  return;
                const h2 = l2.elements.reduce((e4, t5) => "w:ftr" === t5.name ? t5 : e4), p2 = u(h2), d2 = new _i(n2, s2 + i3, p2);
                return yield this.addRelationshipToWrapper(o2, e3, d2, n2), { type: t4.type, footer: d2 };
              })).filter((e4) => !!e4);
              return Promise.all(i2);
            });
          }
          createHeaders(e3, t3, r2, n2, s2) {
            return da(this, void 0, void 0, function* () {
              const i2 = t3.headers.map((t4, i3) => da(this, void 0, void 0, function* () {
                const o2 = r2.find((e4) => e4.id === t4.id);
                if (null === o2 || !o2)
                  throw new Error(`Can not find target file for id ${t4.id}`);
                const a2 = yield e3.files[`word/${o2.target}`].async("text"), l2 = (0, c.xml2js)(a2, { compact: false, captureSpacesBetweenElements: true });
                if (!l2.elements)
                  return;
                const h2 = l2.elements.reduce((e4, t5) => "w:hdr" === t5.name ? t5 : e4), p2 = u(h2), d2 = new Fi(n2, s2 + i3, p2);
                return yield this.addRelationshipToWrapper(o2, e3, d2, n2), { type: t4.type, header: d2 };
              })).filter((e4) => !!e4);
              return Promise.all(i2);
            });
          }
          addRelationshipToWrapper(e3, t3, r2, n2) {
            return da(this, void 0, void 0, function* () {
              const s2 = t3.files[`word/_rels/${e3.target}.rels`];
              if (!s2)
                return;
              const i2 = yield s2.async("text"), o2 = this.findReferenceFiles(i2).filter((e4) => e4.type === ma.IMAGE), a2 = this.findReferenceFiles(i2).filter((e4) => e4.type === ma.HYPERLINK);
              for (const e4 of o2) {
                const s3 = ia.support.arraybuffer ? "arraybuffer" : "nodebuffer", i3 = yield t3.files[`word/${e4.target}`].async(s3), o3 = n2.addMedia(i3, { width: 100, height: 100 });
                r2.Relationships.createRelationship(e4.id, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image", `media/${o3.fileName}`);
              }
              for (const e4 of a2)
                r2.Relationships.createRelationship(e4.id, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", e4.target, Ce.EXTERNAL);
            });
          }
          findReferenceFiles(e3) {
            const t3 = (0, c.xml2js)(e3, { compact: true });
            return (Array.isArray(t3.Relationships.Relationship) ? t3.Relationships.Relationship : [t3.Relationships.Relationship]).map((e4) => {
              if (void 0 === e4._attributes)
                throw Error("relationship element has no attributes");
              return { id: this.parseRefId(e4._attributes.Id), type: fa[e4._attributes.Type], target: e4._attributes.Target };
            }).filter((e4) => null !== e4.type);
          }
          extractDocumentRefs(e3) {
            const t3 = (0, c.xml2js)(e3, { compact: true })["w:document"]["w:body"]["w:sectPr"], r2 = t3["w:headerReference"];
            let n2;
            n2 = void 0 === r2 ? [] : Array.isArray(r2) ? r2 : [r2];
            const s2 = n2.map((e4) => {
              if (void 0 === e4._attributes)
                throw Error("header reference element has no attributes");
              return { type: e4._attributes["w:type"], id: this.parseRefId(e4._attributes["r:id"]) };
            }), i2 = t3["w:footerReference"];
            let o2;
            return o2 = void 0 === i2 ? [] : Array.isArray(i2) ? i2 : [i2], { headers: s2, footers: o2.map((e4) => {
              if (void 0 === e4._attributes)
                throw Error("footer reference element has no attributes");
              return { type: e4._attributes["w:type"], id: this.parseRefId(e4._attributes["r:id"]) };
            }) };
          }
          checkIfTitlePageIsDefined(e3) {
            return void 0 !== (0, c.xml2js)(e3, { compact: true })["w:document"]["w:body"]["w:sectPr"]["w:titlePg"];
          }
          parseRefId(e3) {
            const t3 = /^rId(\d+)$/.exec(e3);
            if (null === t3)
              throw new Error("Invalid ref id");
            return parseInt(t3[1], 10);
          }
        }
        var ga = r(5575);
      })(), n;
    })());
  }
});

// ../../node_modules/buffer-image-size/lib/types/bmp.js
var require_bmp = __commonJS({
  "../../node_modules/buffer-image-size/lib/types/bmp.js"(exports3, module) {
    "use strict";
    function isBMP(buffer) {
      return "BM" === buffer.toString("ascii", 0, 2);
    }
    function calculate(buffer) {
      return {
        "width": buffer.readUInt32LE(18),
        "height": Math.abs(buffer.readInt32LE(22))
      };
    }
    module.exports = {
      "detect": isBMP,
      "calculate": calculate
    };
  }
});

// ../../node_modules/buffer-image-size/lib/types/ico.js
var require_ico = __commonJS({
  "../../node_modules/buffer-image-size/lib/types/ico.js"(exports3, module) {
    "use strict";
    var TYPE_ICON = 1;
    var SIZE_HEADER = 2 + 2 + 2;
    var SIZE_IMAGE_ENTRY = 1 + 1 + 1 + 1 + 2 + 2 + 4 + 4;
    function isICO(buffer) {
      var type;
      if (buffer.readUInt16LE(0) !== 0) {
        return false;
      }
      type = buffer.readUInt16LE(2);
      return type === TYPE_ICON;
    }
    function getSizeFromOffset(buffer, offset) {
      var value = buffer.readUInt8(offset);
      return value === 0 ? 256 : value;
    }
    function getImageSize(buffer, imageIndex) {
      var offset = SIZE_HEADER + imageIndex * SIZE_IMAGE_ENTRY;
      return {
        "width": getSizeFromOffset(buffer, offset),
        "height": getSizeFromOffset(buffer, offset + 1)
      };
    }
    function calculate(buffer) {
      var nbImages = buffer.readUInt16LE(4), result = getImageSize(buffer, 0), imageIndex;
      if (nbImages === 1) {
        return result;
      }
      result.images = [{
        width: result.width,
        height: result.height
      }];
      for (imageIndex = 1; imageIndex < nbImages; imageIndex += 1) {
        result.images.push(getImageSize(buffer, imageIndex));
      }
      return result;
    }
    module.exports = {
      "detect": isICO,
      "calculate": calculate
    };
  }
});

// ../../node_modules/buffer-image-size/lib/types/cur.js
var require_cur = __commonJS({
  "../../node_modules/buffer-image-size/lib/types/cur.js"(exports3, module) {
    "use strict";
    var TYPE_CURSOR = 2;
    function isCUR(buffer) {
      var type;
      if (buffer.readUInt16LE(0) !== 0) {
        return false;
      }
      type = buffer.readUInt16LE(2);
      return type === TYPE_CURSOR;
    }
    module.exports = {
      "detect": isCUR,
      "calculate": require_ico().calculate
    };
  }
});

// ../../node_modules/buffer-image-size/lib/types/dds.js
var require_dds = __commonJS({
  "../../node_modules/buffer-image-size/lib/types/dds.js"(exports3, module) {
    "use strict";
    function isDDS(buffer) {
      return buffer.readUInt32LE(0) === 542327876;
    }
    function calculate(buffer) {
      return {
        "height": buffer.readUInt32LE(12),
        "width": buffer.readUInt32LE(16)
      };
    }
    module.exports = {
      "detect": isDDS,
      "calculate": calculate
    };
  }
});

// ../../node_modules/buffer-image-size/lib/types/gif.js
var require_gif = __commonJS({
  "../../node_modules/buffer-image-size/lib/types/gif.js"(exports3, module) {
    "use strict";
    var gifRegexp = /^GIF8[79]a/;
    function isGIF(buffer) {
      var signature = buffer.toString("ascii", 0, 6);
      return gifRegexp.test(signature);
    }
    function calculate(buffer) {
      return {
        "width": buffer.readUInt16LE(6),
        "height": buffer.readUInt16LE(8)
      };
    }
    module.exports = {
      "detect": isGIF,
      "calculate": calculate
    };
  }
});

// ../../node_modules/buffer-image-size/lib/types/jpg.js
var require_jpg = __commonJS({
  "../../node_modules/buffer-image-size/lib/types/jpg.js"(exports3, module) {
    "use strict";
    function isJPG(buffer) {
      var SOIMarker = buffer.toString("hex", 0, 2);
      return "ffd8" === SOIMarker;
    }
    function extractSize(buffer, i) {
      return {
        "height": buffer.readUInt16BE(i),
        "width": buffer.readUInt16BE(i + 2)
      };
    }
    function validateBuffer(buffer, i) {
      if (i > buffer.length) {
        throw new TypeError("Corrupt JPG, exceeded buffer limits");
      }
      if (buffer[i] !== 255) {
        throw new TypeError("Invalid JPG, marker table corrupted");
      }
    }
    function calculate(buffer) {
      buffer = buffer.slice(4);
      var i, next;
      while (buffer.length) {
        i = buffer.readUInt16BE(0);
        validateBuffer(buffer, i);
        next = buffer[i + 1];
        if (next === 192 || next === 193 || next === 194) {
          return extractSize(buffer, i + 5);
        }
        buffer = buffer.slice(i + 2);
      }
      throw new TypeError("Invalid JPG, no size found");
    }
    module.exports = {
      "detect": isJPG,
      "calculate": calculate
    };
  }
});

// ../../node_modules/buffer-image-size/lib/types/png.js
var require_png = __commonJS({
  "../../node_modules/buffer-image-size/lib/types/png.js"(exports3, module) {
    "use strict";
    var pngSignature = "PNG\r\n\n";
    var pngImageHeaderChunkName = "IHDR";
    var pngFriedChunkName = "CgBI";
    function isPNG(buffer) {
      if (pngSignature === buffer.toString("ascii", 1, 8)) {
        var chunkName = buffer.toString("ascii", 12, 16);
        if (chunkName === pngFriedChunkName) {
          chunkName = buffer.toString("ascii", 28, 32);
        }
        if (chunkName !== pngImageHeaderChunkName) {
          throw new TypeError("invalid png");
        }
        return true;
      }
    }
    function calculate(buffer) {
      if (buffer.toString("ascii", 12, 16) === pngFriedChunkName) {
        return {
          "width": buffer.readUInt32BE(32),
          "height": buffer.readUInt32BE(36)
        };
      }
      return {
        "width": buffer.readUInt32BE(16),
        "height": buffer.readUInt32BE(20)
      };
    }
    module.exports = {
      "detect": isPNG,
      "calculate": calculate
    };
  }
});

// ../../node_modules/buffer-image-size/lib/types/psd.js
var require_psd = __commonJS({
  "../../node_modules/buffer-image-size/lib/types/psd.js"(exports3, module) {
    "use strict";
    function isPSD(buffer) {
      return "8BPS" === buffer.toString("ascii", 0, 4);
    }
    function calculate(buffer) {
      return {
        "width": buffer.readUInt32BE(18),
        "height": buffer.readUInt32BE(14)
      };
    }
    module.exports = {
      "detect": isPSD,
      "calculate": calculate
    };
  }
});

// ../../node_modules/buffer-image-size/lib/types/svg.js
var require_svg = __commonJS({
  "../../node_modules/buffer-image-size/lib/types/svg.js"(exports3, module) {
    "use strict";
    var svgReg = /<svg[^>]+[^>]*>/;
    function isSVG(buffer) {
      return svgReg.test(buffer);
    }
    var extractorRegExps = {
      "root": /<svg\s[^>]+>/,
      "width": /\bwidth=(['"])([^%]+?)\1/,
      "height": /\bheight=(['"])([^%]+?)\1/,
      "viewbox": /\bviewBox=(['"])(.+?)\1/
    };
    function parseViewbox(viewbox) {
      var bounds = viewbox.split(" ");
      return {
        "width": parseInt(bounds[2], 10),
        "height": parseInt(bounds[3], 10)
      };
    }
    function parseAttributes(root) {
      var width = root.match(extractorRegExps.width);
      var height = root.match(extractorRegExps.height);
      var viewbox = root.match(extractorRegExps.viewbox);
      return {
        "width": width && parseInt(width[2], 10),
        "height": height && parseInt(height[2], 10),
        "viewbox": viewbox && parseViewbox(viewbox[2])
      };
    }
    function calculateByDimensions(attrs) {
      return {
        "width": attrs.width,
        "height": attrs.height
      };
    }
    function calculateByViewbox(attrs) {
      var ratio = attrs.viewbox.width / attrs.viewbox.height;
      if (attrs.width) {
        return {
          "width": attrs.width,
          "height": Math.floor(attrs.width / ratio)
        };
      }
      if (attrs.height) {
        return {
          "width": Math.floor(attrs.height * ratio),
          "height": attrs.height
        };
      }
      return {
        "width": attrs.viewbox.width,
        "height": attrs.viewbox.height
      };
    }
    function calculate(buffer) {
      var root = buffer.toString("utf8").match(extractorRegExps.root);
      if (root) {
        var attrs = parseAttributes(root[0]);
        if (attrs.width && attrs.height) {
          return calculateByDimensions(attrs);
        }
        if (attrs.viewbox) {
          return calculateByViewbox(attrs);
        }
      }
      throw new TypeError("invalid svg");
    }
    module.exports = {
      "detect": isSVG,
      "calculate": calculate
    };
  }
});

// ../../node_modules/buffer-image-size/lib/types/webp.js
var require_webp = __commonJS({
  "../../node_modules/buffer-image-size/lib/types/webp.js"(exports3, module) {
    "use strict";
    function isWebP(buffer) {
      var riffHeader = "RIFF" === buffer.toString("ascii", 0, 4);
      var webpHeader = "WEBP" === buffer.toString("ascii", 8, 12);
      var vp8Header = "VP8" === buffer.toString("ascii", 12, 15);
      return riffHeader && webpHeader && vp8Header;
    }
    function calculate(buffer) {
      var chunkHeader = buffer.toString("ascii", 12, 16);
      buffer = buffer.slice(20, 30);
      if (chunkHeader === "VP8X") {
        var extendedHeader = buffer[0];
        var validStart = (extendedHeader & 192) === 0;
        var validEnd = (extendedHeader & 1) === 0;
        if (validStart && validEnd) {
          return calculateExtended(buffer);
        } else {
          return false;
        }
      }
      if (chunkHeader === "VP8 " && buffer[0] !== 47) {
        return calculateLossy(buffer);
      }
      var signature = buffer.toString("hex", 3, 6);
      if (chunkHeader === "VP8L" && signature !== "9d012a") {
        return calculateLossless(buffer);
      }
      return false;
    }
    function calculateExtended(buffer) {
      return {
        "width": 1 + buffer.readUIntLE(4, 3),
        "height": 1 + buffer.readUIntLE(7, 3)
      };
    }
    function calculateLossless(buffer) {
      return {
        "width": 1 + ((buffer[2] & 63) << 8 | buffer[1]),
        "height": 1 + ((buffer[4] & 15) << 10 | buffer[3] << 2 | (buffer[2] & 192) >> 6)
      };
    }
    function calculateLossy(buffer) {
      return {
        "width": buffer.readInt16LE(6) & 16383,
        "height": buffer.readInt16LE(8) & 16383
      };
    }
    module.exports = {
      "detect": isWebP,
      "calculate": calculate
    };
  }
});

// ../../node_modules/buffer-image-size/lib/types.js
var require_types = __commonJS({
  "../../node_modules/buffer-image-size/lib/types.js"(exports3, module) {
    "use strict";
    var typeHandlers = {
      bmp: require_bmp(),
      cur: require_cur(),
      dds: require_dds(),
      gif: require_gif(),
      ico: require_ico(),
      jpg: require_jpg(),
      png: require_png(),
      psd: require_psd(),
      svg: require_svg(),
      webp: require_webp()
    };
    module.exports = typeHandlers;
  }
});

// ../../node_modules/buffer-image-size/lib/detector.js
var require_detector = __commonJS({
  "../../node_modules/buffer-image-size/lib/detector.js"(exports3, module) {
    "use strict";
    var typeHandlers = require_types();
    module.exports = function(buffer) {
      var type, result;
      for (type in typeHandlers) {
        result = typeHandlers[type].detect(buffer);
        if (result) {
          return type;
        }
      }
    };
  }
});

// ../../node_modules/buffer-image-size/lib/index.js
var require_lib = __commonJS({
  "../../node_modules/buffer-image-size/lib/index.js"(exports3, module) {
    "use strict";
    var typeHandlers = require_types();
    var detector = require_detector();
    function lookup(buffer) {
      var type = detector(buffer);
      if (type in typeHandlers) {
        var size = typeHandlers[type].calculate(buffer);
        if (size !== false) {
          size.type = type;
          return size;
        }
      }
      throw new TypeError("unsupported file type: " + type);
    }
    module.exports = function(input) {
      if (Buffer.isBuffer(input)) {
        return lookup(input);
      }
      throw new TypeError("expecting only a buffer as input");
    };
    module.exports.types = Object.keys(typeHandlers);
  }
});

// ../../node_modules/myst-to-docx/dist/serializer.js
var import_docx4 = __toESM(require_build(), 1);

// ../../node_modules/myst-to-docx/dist/schema.js
var import_docx3 = __toESM(require_build(), 1);

// ../../node_modules/myst-to-docx/dist/utils.js
var import_docx = __toESM(require_build(), 1);

// node-modules-polyfills:buffer
var exports$2 = {};
var _dewExec$2 = false;
function dew$2() {
  if (_dewExec$2)
    return exports$2;
  _dewExec$2 = true;
  exports$2.byteLength = byteLength;
  exports$2.toByteArray = toByteArray;
  exports$2.fromByteArray = fromByteArray;
  var lookup = [];
  var revLookup = [];
  var Arr = typeof Uint8Array !== "undefined" ? Uint8Array : Array;
  var code2 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  for (var i = 0, len = code2.length; i < len; ++i) {
    lookup[i] = code2[i];
    revLookup[code2.charCodeAt(i)] = i;
  }
  revLookup["-".charCodeAt(0)] = 62;
  revLookup["_".charCodeAt(0)] = 63;
  function getLens(b64) {
    var len2 = b64.length;
    if (len2 % 4 > 0) {
      throw new Error("Invalid string. Length must be a multiple of 4");
    }
    var validLen = b64.indexOf("=");
    if (validLen === -1)
      validLen = len2;
    var placeHoldersLen = validLen === len2 ? 0 : 4 - validLen % 4;
    return [validLen, placeHoldersLen];
  }
  function byteLength(b64) {
    var lens = getLens(b64);
    var validLen = lens[0];
    var placeHoldersLen = lens[1];
    return (validLen + placeHoldersLen) * 3 / 4 - placeHoldersLen;
  }
  function _byteLength(b64, validLen, placeHoldersLen) {
    return (validLen + placeHoldersLen) * 3 / 4 - placeHoldersLen;
  }
  function toByteArray(b64) {
    var tmp;
    var lens = getLens(b64);
    var validLen = lens[0];
    var placeHoldersLen = lens[1];
    var arr = new Arr(_byteLength(b64, validLen, placeHoldersLen));
    var curByte = 0;
    var len2 = placeHoldersLen > 0 ? validLen - 4 : validLen;
    var i2;
    for (i2 = 0; i2 < len2; i2 += 4) {
      tmp = revLookup[b64.charCodeAt(i2)] << 18 | revLookup[b64.charCodeAt(i2 + 1)] << 12 | revLookup[b64.charCodeAt(i2 + 2)] << 6 | revLookup[b64.charCodeAt(i2 + 3)];
      arr[curByte++] = tmp >> 16 & 255;
      arr[curByte++] = tmp >> 8 & 255;
      arr[curByte++] = tmp & 255;
    }
    if (placeHoldersLen === 2) {
      tmp = revLookup[b64.charCodeAt(i2)] << 2 | revLookup[b64.charCodeAt(i2 + 1)] >> 4;
      arr[curByte++] = tmp & 255;
    }
    if (placeHoldersLen === 1) {
      tmp = revLookup[b64.charCodeAt(i2)] << 10 | revLookup[b64.charCodeAt(i2 + 1)] << 4 | revLookup[b64.charCodeAt(i2 + 2)] >> 2;
      arr[curByte++] = tmp >> 8 & 255;
      arr[curByte++] = tmp & 255;
    }
    return arr;
  }
  function tripletToBase64(num) {
    return lookup[num >> 18 & 63] + lookup[num >> 12 & 63] + lookup[num >> 6 & 63] + lookup[num & 63];
  }
  function encodeChunk(uint8, start, end) {
    var tmp;
    var output = [];
    for (var i2 = start; i2 < end; i2 += 3) {
      tmp = (uint8[i2] << 16 & 16711680) + (uint8[i2 + 1] << 8 & 65280) + (uint8[i2 + 2] & 255);
      output.push(tripletToBase64(tmp));
    }
    return output.join("");
  }
  function fromByteArray(uint8) {
    var tmp;
    var len2 = uint8.length;
    var extraBytes = len2 % 3;
    var parts = [];
    var maxChunkLength = 16383;
    for (var i2 = 0, len22 = len2 - extraBytes; i2 < len22; i2 += maxChunkLength) {
      parts.push(encodeChunk(uint8, i2, i2 + maxChunkLength > len22 ? len22 : i2 + maxChunkLength));
    }
    if (extraBytes === 1) {
      tmp = uint8[len2 - 1];
      parts.push(lookup[tmp >> 2] + lookup[tmp << 4 & 63] + "==");
    } else if (extraBytes === 2) {
      tmp = (uint8[len2 - 2] << 8) + uint8[len2 - 1];
      parts.push(lookup[tmp >> 10] + lookup[tmp >> 4 & 63] + lookup[tmp << 2 & 63] + "=");
    }
    return parts.join("");
  }
  return exports$2;
}
var exports$1 = {};
var _dewExec$1 = false;
function dew$1() {
  if (_dewExec$1)
    return exports$1;
  _dewExec$1 = true;
  exports$1.read = function(buffer, offset, isLE, mLen, nBytes) {
    var e, m;
    var eLen = nBytes * 8 - mLen - 1;
    var eMax = (1 << eLen) - 1;
    var eBias = eMax >> 1;
    var nBits = -7;
    var i = isLE ? nBytes - 1 : 0;
    var d = isLE ? -1 : 1;
    var s = buffer[offset + i];
    i += d;
    e = s & (1 << -nBits) - 1;
    s >>= -nBits;
    nBits += eLen;
    for (; nBits > 0; e = e * 256 + buffer[offset + i], i += d, nBits -= 8) {
    }
    m = e & (1 << -nBits) - 1;
    e >>= -nBits;
    nBits += mLen;
    for (; nBits > 0; m = m * 256 + buffer[offset + i], i += d, nBits -= 8) {
    }
    if (e === 0) {
      e = 1 - eBias;
    } else if (e === eMax) {
      return m ? NaN : (s ? -1 : 1) * Infinity;
    } else {
      m = m + Math.pow(2, mLen);
      e = e - eBias;
    }
    return (s ? -1 : 1) * m * Math.pow(2, e - mLen);
  };
  exports$1.write = function(buffer, value, offset, isLE, mLen, nBytes) {
    var e, m, c;
    var eLen = nBytes * 8 - mLen - 1;
    var eMax = (1 << eLen) - 1;
    var eBias = eMax >> 1;
    var rt = mLen === 23 ? Math.pow(2, -24) - Math.pow(2, -77) : 0;
    var i = isLE ? 0 : nBytes - 1;
    var d = isLE ? 1 : -1;
    var s = value < 0 || value === 0 && 1 / value < 0 ? 1 : 0;
    value = Math.abs(value);
    if (isNaN(value) || value === Infinity) {
      m = isNaN(value) ? 1 : 0;
      e = eMax;
    } else {
      e = Math.floor(Math.log(value) / Math.LN2);
      if (value * (c = Math.pow(2, -e)) < 1) {
        e--;
        c *= 2;
      }
      if (e + eBias >= 1) {
        value += rt / c;
      } else {
        value += rt * Math.pow(2, 1 - eBias);
      }
      if (value * c >= 2) {
        e++;
        c /= 2;
      }
      if (e + eBias >= eMax) {
        m = 0;
        e = eMax;
      } else if (e + eBias >= 1) {
        m = (value * c - 1) * Math.pow(2, mLen);
        e = e + eBias;
      } else {
        m = value * Math.pow(2, eBias - 1) * Math.pow(2, mLen);
        e = 0;
      }
    }
    for (; mLen >= 8; buffer[offset + i] = m & 255, i += d, m /= 256, mLen -= 8) {
    }
    e = e << mLen | m;
    eLen += mLen;
    for (; eLen > 0; buffer[offset + i] = e & 255, i += d, e /= 256, eLen -= 8) {
    }
    buffer[offset + i - d] |= s * 128;
  };
  return exports$1;
}
var exports = {};
var _dewExec = false;
function dew() {
  if (_dewExec)
    return exports;
  _dewExec = true;
  const base64 = dew$2();
  const ieee754 = dew$1();
  const customInspectSymbol = typeof Symbol === "function" && typeof Symbol["for"] === "function" ? Symbol["for"]("nodejs.util.inspect.custom") : null;
  exports.Buffer = Buffer22;
  exports.SlowBuffer = SlowBuffer;
  exports.INSPECT_MAX_BYTES = 50;
  const K_MAX_LENGTH = 2147483647;
  exports.kMaxLength = K_MAX_LENGTH;
  Buffer22.TYPED_ARRAY_SUPPORT = typedArraySupport();
  if (!Buffer22.TYPED_ARRAY_SUPPORT && typeof console !== "undefined" && typeof console.error === "function") {
    console.error("This browser lacks typed array (Uint8Array) support which is required by `buffer` v5.x. Use `buffer` v4.x if you require old browser support.");
  }
  function typedArraySupport() {
    try {
      const arr = new Uint8Array(1);
      const proto = {
        foo: function() {
          return 42;
        }
      };
      Object.setPrototypeOf(proto, Uint8Array.prototype);
      Object.setPrototypeOf(arr, proto);
      return arr.foo() === 42;
    } catch (e) {
      return false;
    }
  }
  Object.defineProperty(Buffer22.prototype, "parent", {
    enumerable: true,
    get: function() {
      if (!Buffer22.isBuffer(this))
        return void 0;
      return this.buffer;
    }
  });
  Object.defineProperty(Buffer22.prototype, "offset", {
    enumerable: true,
    get: function() {
      if (!Buffer22.isBuffer(this))
        return void 0;
      return this.byteOffset;
    }
  });
  function createBuffer(length) {
    if (length > K_MAX_LENGTH) {
      throw new RangeError('The value "' + length + '" is invalid for option "size"');
    }
    const buf = new Uint8Array(length);
    Object.setPrototypeOf(buf, Buffer22.prototype);
    return buf;
  }
  function Buffer22(arg, encodingOrOffset, length) {
    if (typeof arg === "number") {
      if (typeof encodingOrOffset === "string") {
        throw new TypeError('The "string" argument must be of type string. Received type number');
      }
      return allocUnsafe(arg);
    }
    return from(arg, encodingOrOffset, length);
  }
  Buffer22.poolSize = 8192;
  function from(value, encodingOrOffset, length) {
    if (typeof value === "string") {
      return fromString(value, encodingOrOffset);
    }
    if (ArrayBuffer.isView(value)) {
      return fromArrayView(value);
    }
    if (value == null) {
      throw new TypeError("The first argument must be one of type string, Buffer, ArrayBuffer, Array, or Array-like Object. Received type " + typeof value);
    }
    if (isInstance(value, ArrayBuffer) || value && isInstance(value.buffer, ArrayBuffer)) {
      return fromArrayBuffer(value, encodingOrOffset, length);
    }
    if (typeof SharedArrayBuffer !== "undefined" && (isInstance(value, SharedArrayBuffer) || value && isInstance(value.buffer, SharedArrayBuffer))) {
      return fromArrayBuffer(value, encodingOrOffset, length);
    }
    if (typeof value === "number") {
      throw new TypeError('The "value" argument must not be of type number. Received type number');
    }
    const valueOf = value.valueOf && value.valueOf();
    if (valueOf != null && valueOf !== value) {
      return Buffer22.from(valueOf, encodingOrOffset, length);
    }
    const b = fromObject(value);
    if (b)
      return b;
    if (typeof Symbol !== "undefined" && Symbol.toPrimitive != null && typeof value[Symbol.toPrimitive] === "function") {
      return Buffer22.from(value[Symbol.toPrimitive]("string"), encodingOrOffset, length);
    }
    throw new TypeError("The first argument must be one of type string, Buffer, ArrayBuffer, Array, or Array-like Object. Received type " + typeof value);
  }
  Buffer22.from = function(value, encodingOrOffset, length) {
    return from(value, encodingOrOffset, length);
  };
  Object.setPrototypeOf(Buffer22.prototype, Uint8Array.prototype);
  Object.setPrototypeOf(Buffer22, Uint8Array);
  function assertSize(size) {
    if (typeof size !== "number") {
      throw new TypeError('"size" argument must be of type number');
    } else if (size < 0) {
      throw new RangeError('The value "' + size + '" is invalid for option "size"');
    }
  }
  function alloc(size, fill, encoding) {
    assertSize(size);
    if (size <= 0) {
      return createBuffer(size);
    }
    if (fill !== void 0) {
      return typeof encoding === "string" ? createBuffer(size).fill(fill, encoding) : createBuffer(size).fill(fill);
    }
    return createBuffer(size);
  }
  Buffer22.alloc = function(size, fill, encoding) {
    return alloc(size, fill, encoding);
  };
  function allocUnsafe(size) {
    assertSize(size);
    return createBuffer(size < 0 ? 0 : checked(size) | 0);
  }
  Buffer22.allocUnsafe = function(size) {
    return allocUnsafe(size);
  };
  Buffer22.allocUnsafeSlow = function(size) {
    return allocUnsafe(size);
  };
  function fromString(string, encoding) {
    if (typeof encoding !== "string" || encoding === "") {
      encoding = "utf8";
    }
    if (!Buffer22.isEncoding(encoding)) {
      throw new TypeError("Unknown encoding: " + encoding);
    }
    const length = byteLength(string, encoding) | 0;
    let buf = createBuffer(length);
    const actual = buf.write(string, encoding);
    if (actual !== length) {
      buf = buf.slice(0, actual);
    }
    return buf;
  }
  function fromArrayLike(array) {
    const length = array.length < 0 ? 0 : checked(array.length) | 0;
    const buf = createBuffer(length);
    for (let i = 0; i < length; i += 1) {
      buf[i] = array[i] & 255;
    }
    return buf;
  }
  function fromArrayView(arrayView) {
    if (isInstance(arrayView, Uint8Array)) {
      const copy = new Uint8Array(arrayView);
      return fromArrayBuffer(copy.buffer, copy.byteOffset, copy.byteLength);
    }
    return fromArrayLike(arrayView);
  }
  function fromArrayBuffer(array, byteOffset, length) {
    if (byteOffset < 0 || array.byteLength < byteOffset) {
      throw new RangeError('"offset" is outside of buffer bounds');
    }
    if (array.byteLength < byteOffset + (length || 0)) {
      throw new RangeError('"length" is outside of buffer bounds');
    }
    let buf;
    if (byteOffset === void 0 && length === void 0) {
      buf = new Uint8Array(array);
    } else if (length === void 0) {
      buf = new Uint8Array(array, byteOffset);
    } else {
      buf = new Uint8Array(array, byteOffset, length);
    }
    Object.setPrototypeOf(buf, Buffer22.prototype);
    return buf;
  }
  function fromObject(obj) {
    if (Buffer22.isBuffer(obj)) {
      const len = checked(obj.length) | 0;
      const buf = createBuffer(len);
      if (buf.length === 0) {
        return buf;
      }
      obj.copy(buf, 0, 0, len);
      return buf;
    }
    if (obj.length !== void 0) {
      if (typeof obj.length !== "number" || numberIsNaN(obj.length)) {
        return createBuffer(0);
      }
      return fromArrayLike(obj);
    }
    if (obj.type === "Buffer" && Array.isArray(obj.data)) {
      return fromArrayLike(obj.data);
    }
  }
  function checked(length) {
    if (length >= K_MAX_LENGTH) {
      throw new RangeError("Attempt to allocate Buffer larger than maximum size: 0x" + K_MAX_LENGTH.toString(16) + " bytes");
    }
    return length | 0;
  }
  function SlowBuffer(length) {
    if (+length != length) {
      length = 0;
    }
    return Buffer22.alloc(+length);
  }
  Buffer22.isBuffer = function isBuffer(b) {
    return b != null && b._isBuffer === true && b !== Buffer22.prototype;
  };
  Buffer22.compare = function compare(a, b) {
    if (isInstance(a, Uint8Array))
      a = Buffer22.from(a, a.offset, a.byteLength);
    if (isInstance(b, Uint8Array))
      b = Buffer22.from(b, b.offset, b.byteLength);
    if (!Buffer22.isBuffer(a) || !Buffer22.isBuffer(b)) {
      throw new TypeError('The "buf1", "buf2" arguments must be one of type Buffer or Uint8Array');
    }
    if (a === b)
      return 0;
    let x = a.length;
    let y = b.length;
    for (let i = 0, len = Math.min(x, y); i < len; ++i) {
      if (a[i] !== b[i]) {
        x = a[i];
        y = b[i];
        break;
      }
    }
    if (x < y)
      return -1;
    if (y < x)
      return 1;
    return 0;
  };
  Buffer22.isEncoding = function isEncoding(encoding) {
    switch (String(encoding).toLowerCase()) {
      case "hex":
      case "utf8":
      case "utf-8":
      case "ascii":
      case "latin1":
      case "binary":
      case "base64":
      case "ucs2":
      case "ucs-2":
      case "utf16le":
      case "utf-16le":
        return true;
      default:
        return false;
    }
  };
  Buffer22.concat = function concat(list2, length) {
    if (!Array.isArray(list2)) {
      throw new TypeError('"list" argument must be an Array of Buffers');
    }
    if (list2.length === 0) {
      return Buffer22.alloc(0);
    }
    let i;
    if (length === void 0) {
      length = 0;
      for (i = 0; i < list2.length; ++i) {
        length += list2[i].length;
      }
    }
    const buffer = Buffer22.allocUnsafe(length);
    let pos = 0;
    for (i = 0; i < list2.length; ++i) {
      let buf = list2[i];
      if (isInstance(buf, Uint8Array)) {
        if (pos + buf.length > buffer.length) {
          if (!Buffer22.isBuffer(buf))
            buf = Buffer22.from(buf);
          buf.copy(buffer, pos);
        } else {
          Uint8Array.prototype.set.call(buffer, buf, pos);
        }
      } else if (!Buffer22.isBuffer(buf)) {
        throw new TypeError('"list" argument must be an Array of Buffers');
      } else {
        buf.copy(buffer, pos);
      }
      pos += buf.length;
    }
    return buffer;
  };
  function byteLength(string, encoding) {
    if (Buffer22.isBuffer(string)) {
      return string.length;
    }
    if (ArrayBuffer.isView(string) || isInstance(string, ArrayBuffer)) {
      return string.byteLength;
    }
    if (typeof string !== "string") {
      throw new TypeError('The "string" argument must be one of type string, Buffer, or ArrayBuffer. Received type ' + typeof string);
    }
    const len = string.length;
    const mustMatch = arguments.length > 2 && arguments[2] === true;
    if (!mustMatch && len === 0)
      return 0;
    let loweredCase = false;
    for (; ; ) {
      switch (encoding) {
        case "ascii":
        case "latin1":
        case "binary":
          return len;
        case "utf8":
        case "utf-8":
          return utf8ToBytes(string).length;
        case "ucs2":
        case "ucs-2":
        case "utf16le":
        case "utf-16le":
          return len * 2;
        case "hex":
          return len >>> 1;
        case "base64":
          return base64ToBytes(string).length;
        default:
          if (loweredCase) {
            return mustMatch ? -1 : utf8ToBytes(string).length;
          }
          encoding = ("" + encoding).toLowerCase();
          loweredCase = true;
      }
    }
  }
  Buffer22.byteLength = byteLength;
  function slowToString(encoding, start, end) {
    let loweredCase = false;
    if (start === void 0 || start < 0) {
      start = 0;
    }
    if (start > this.length) {
      return "";
    }
    if (end === void 0 || end > this.length) {
      end = this.length;
    }
    if (end <= 0) {
      return "";
    }
    end >>>= 0;
    start >>>= 0;
    if (end <= start) {
      return "";
    }
    if (!encoding)
      encoding = "utf8";
    while (true) {
      switch (encoding) {
        case "hex":
          return hexSlice(this, start, end);
        case "utf8":
        case "utf-8":
          return utf8Slice(this, start, end);
        case "ascii":
          return asciiSlice(this, start, end);
        case "latin1":
        case "binary":
          return latin1Slice(this, start, end);
        case "base64":
          return base64Slice(this, start, end);
        case "ucs2":
        case "ucs-2":
        case "utf16le":
        case "utf-16le":
          return utf16leSlice(this, start, end);
        default:
          if (loweredCase)
            throw new TypeError("Unknown encoding: " + encoding);
          encoding = (encoding + "").toLowerCase();
          loweredCase = true;
      }
    }
  }
  Buffer22.prototype._isBuffer = true;
  function swap(b, n, m) {
    const i = b[n];
    b[n] = b[m];
    b[m] = i;
  }
  Buffer22.prototype.swap16 = function swap16() {
    const len = this.length;
    if (len % 2 !== 0) {
      throw new RangeError("Buffer size must be a multiple of 16-bits");
    }
    for (let i = 0; i < len; i += 2) {
      swap(this, i, i + 1);
    }
    return this;
  };
  Buffer22.prototype.swap32 = function swap32() {
    const len = this.length;
    if (len % 4 !== 0) {
      throw new RangeError("Buffer size must be a multiple of 32-bits");
    }
    for (let i = 0; i < len; i += 4) {
      swap(this, i, i + 3);
      swap(this, i + 1, i + 2);
    }
    return this;
  };
  Buffer22.prototype.swap64 = function swap64() {
    const len = this.length;
    if (len % 8 !== 0) {
      throw new RangeError("Buffer size must be a multiple of 64-bits");
    }
    for (let i = 0; i < len; i += 8) {
      swap(this, i, i + 7);
      swap(this, i + 1, i + 6);
      swap(this, i + 2, i + 5);
      swap(this, i + 3, i + 4);
    }
    return this;
  };
  Buffer22.prototype.toString = function toString() {
    const length = this.length;
    if (length === 0)
      return "";
    if (arguments.length === 0)
      return utf8Slice(this, 0, length);
    return slowToString.apply(this, arguments);
  };
  Buffer22.prototype.toLocaleString = Buffer22.prototype.toString;
  Buffer22.prototype.equals = function equals(b) {
    if (!Buffer22.isBuffer(b))
      throw new TypeError("Argument must be a Buffer");
    if (this === b)
      return true;
    return Buffer22.compare(this, b) === 0;
  };
  Buffer22.prototype.inspect = function inspect() {
    let str = "";
    const max = exports.INSPECT_MAX_BYTES;
    str = this.toString("hex", 0, max).replace(/(.{2})/g, "$1 ").trim();
    if (this.length > max)
      str += " ... ";
    return "<Buffer " + str + ">";
  };
  if (customInspectSymbol) {
    Buffer22.prototype[customInspectSymbol] = Buffer22.prototype.inspect;
  }
  Buffer22.prototype.compare = function compare(target, start, end, thisStart, thisEnd) {
    if (isInstance(target, Uint8Array)) {
      target = Buffer22.from(target, target.offset, target.byteLength);
    }
    if (!Buffer22.isBuffer(target)) {
      throw new TypeError('The "target" argument must be one of type Buffer or Uint8Array. Received type ' + typeof target);
    }
    if (start === void 0) {
      start = 0;
    }
    if (end === void 0) {
      end = target ? target.length : 0;
    }
    if (thisStart === void 0) {
      thisStart = 0;
    }
    if (thisEnd === void 0) {
      thisEnd = this.length;
    }
    if (start < 0 || end > target.length || thisStart < 0 || thisEnd > this.length) {
      throw new RangeError("out of range index");
    }
    if (thisStart >= thisEnd && start >= end) {
      return 0;
    }
    if (thisStart >= thisEnd) {
      return -1;
    }
    if (start >= end) {
      return 1;
    }
    start >>>= 0;
    end >>>= 0;
    thisStart >>>= 0;
    thisEnd >>>= 0;
    if (this === target)
      return 0;
    let x = thisEnd - thisStart;
    let y = end - start;
    const len = Math.min(x, y);
    const thisCopy = this.slice(thisStart, thisEnd);
    const targetCopy = target.slice(start, end);
    for (let i = 0; i < len; ++i) {
      if (thisCopy[i] !== targetCopy[i]) {
        x = thisCopy[i];
        y = targetCopy[i];
        break;
      }
    }
    if (x < y)
      return -1;
    if (y < x)
      return 1;
    return 0;
  };
  function bidirectionalIndexOf(buffer, val, byteOffset, encoding, dir) {
    if (buffer.length === 0)
      return -1;
    if (typeof byteOffset === "string") {
      encoding = byteOffset;
      byteOffset = 0;
    } else if (byteOffset > 2147483647) {
      byteOffset = 2147483647;
    } else if (byteOffset < -2147483648) {
      byteOffset = -2147483648;
    }
    byteOffset = +byteOffset;
    if (numberIsNaN(byteOffset)) {
      byteOffset = dir ? 0 : buffer.length - 1;
    }
    if (byteOffset < 0)
      byteOffset = buffer.length + byteOffset;
    if (byteOffset >= buffer.length) {
      if (dir)
        return -1;
      else
        byteOffset = buffer.length - 1;
    } else if (byteOffset < 0) {
      if (dir)
        byteOffset = 0;
      else
        return -1;
    }
    if (typeof val === "string") {
      val = Buffer22.from(val, encoding);
    }
    if (Buffer22.isBuffer(val)) {
      if (val.length === 0) {
        return -1;
      }
      return arrayIndexOf(buffer, val, byteOffset, encoding, dir);
    } else if (typeof val === "number") {
      val = val & 255;
      if (typeof Uint8Array.prototype.indexOf === "function") {
        if (dir) {
          return Uint8Array.prototype.indexOf.call(buffer, val, byteOffset);
        } else {
          return Uint8Array.prototype.lastIndexOf.call(buffer, val, byteOffset);
        }
      }
      return arrayIndexOf(buffer, [val], byteOffset, encoding, dir);
    }
    throw new TypeError("val must be string, number or Buffer");
  }
  function arrayIndexOf(arr, val, byteOffset, encoding, dir) {
    let indexSize = 1;
    let arrLength = arr.length;
    let valLength = val.length;
    if (encoding !== void 0) {
      encoding = String(encoding).toLowerCase();
      if (encoding === "ucs2" || encoding === "ucs-2" || encoding === "utf16le" || encoding === "utf-16le") {
        if (arr.length < 2 || val.length < 2) {
          return -1;
        }
        indexSize = 2;
        arrLength /= 2;
        valLength /= 2;
        byteOffset /= 2;
      }
    }
    function read(buf, i2) {
      if (indexSize === 1) {
        return buf[i2];
      } else {
        return buf.readUInt16BE(i2 * indexSize);
      }
    }
    let i;
    if (dir) {
      let foundIndex = -1;
      for (i = byteOffset; i < arrLength; i++) {
        if (read(arr, i) === read(val, foundIndex === -1 ? 0 : i - foundIndex)) {
          if (foundIndex === -1)
            foundIndex = i;
          if (i - foundIndex + 1 === valLength)
            return foundIndex * indexSize;
        } else {
          if (foundIndex !== -1)
            i -= i - foundIndex;
          foundIndex = -1;
        }
      }
    } else {
      if (byteOffset + valLength > arrLength)
        byteOffset = arrLength - valLength;
      for (i = byteOffset; i >= 0; i--) {
        let found = true;
        for (let j = 0; j < valLength; j++) {
          if (read(arr, i + j) !== read(val, j)) {
            found = false;
            break;
          }
        }
        if (found)
          return i;
      }
    }
    return -1;
  }
  Buffer22.prototype.includes = function includes(val, byteOffset, encoding) {
    return this.indexOf(val, byteOffset, encoding) !== -1;
  };
  Buffer22.prototype.indexOf = function indexOf(val, byteOffset, encoding) {
    return bidirectionalIndexOf(this, val, byteOffset, encoding, true);
  };
  Buffer22.prototype.lastIndexOf = function lastIndexOf(val, byteOffset, encoding) {
    return bidirectionalIndexOf(this, val, byteOffset, encoding, false);
  };
  function hexWrite(buf, string, offset, length) {
    offset = Number(offset) || 0;
    const remaining = buf.length - offset;
    if (!length) {
      length = remaining;
    } else {
      length = Number(length);
      if (length > remaining) {
        length = remaining;
      }
    }
    const strLen = string.length;
    if (length > strLen / 2) {
      length = strLen / 2;
    }
    let i;
    for (i = 0; i < length; ++i) {
      const parsed = parseInt(string.substr(i * 2, 2), 16);
      if (numberIsNaN(parsed))
        return i;
      buf[offset + i] = parsed;
    }
    return i;
  }
  function utf8Write(buf, string, offset, length) {
    return blitBuffer(utf8ToBytes(string, buf.length - offset), buf, offset, length);
  }
  function asciiWrite(buf, string, offset, length) {
    return blitBuffer(asciiToBytes(string), buf, offset, length);
  }
  function base64Write(buf, string, offset, length) {
    return blitBuffer(base64ToBytes(string), buf, offset, length);
  }
  function ucs2Write(buf, string, offset, length) {
    return blitBuffer(utf16leToBytes(string, buf.length - offset), buf, offset, length);
  }
  Buffer22.prototype.write = function write(string, offset, length, encoding) {
    if (offset === void 0) {
      encoding = "utf8";
      length = this.length;
      offset = 0;
    } else if (length === void 0 && typeof offset === "string") {
      encoding = offset;
      length = this.length;
      offset = 0;
    } else if (isFinite(offset)) {
      offset = offset >>> 0;
      if (isFinite(length)) {
        length = length >>> 0;
        if (encoding === void 0)
          encoding = "utf8";
      } else {
        encoding = length;
        length = void 0;
      }
    } else {
      throw new Error("Buffer.write(string, encoding, offset[, length]) is no longer supported");
    }
    const remaining = this.length - offset;
    if (length === void 0 || length > remaining)
      length = remaining;
    if (string.length > 0 && (length < 0 || offset < 0) || offset > this.length) {
      throw new RangeError("Attempt to write outside buffer bounds");
    }
    if (!encoding)
      encoding = "utf8";
    let loweredCase = false;
    for (; ; ) {
      switch (encoding) {
        case "hex":
          return hexWrite(this, string, offset, length);
        case "utf8":
        case "utf-8":
          return utf8Write(this, string, offset, length);
        case "ascii":
        case "latin1":
        case "binary":
          return asciiWrite(this, string, offset, length);
        case "base64":
          return base64Write(this, string, offset, length);
        case "ucs2":
        case "ucs-2":
        case "utf16le":
        case "utf-16le":
          return ucs2Write(this, string, offset, length);
        default:
          if (loweredCase)
            throw new TypeError("Unknown encoding: " + encoding);
          encoding = ("" + encoding).toLowerCase();
          loweredCase = true;
      }
    }
  };
  Buffer22.prototype.toJSON = function toJSON() {
    return {
      type: "Buffer",
      data: Array.prototype.slice.call(this._arr || this, 0)
    };
  };
  function base64Slice(buf, start, end) {
    if (start === 0 && end === buf.length) {
      return base64.fromByteArray(buf);
    } else {
      return base64.fromByteArray(buf.slice(start, end));
    }
  }
  function utf8Slice(buf, start, end) {
    end = Math.min(buf.length, end);
    const res = [];
    let i = start;
    while (i < end) {
      const firstByte = buf[i];
      let codePoint = null;
      let bytesPerSequence = firstByte > 239 ? 4 : firstByte > 223 ? 3 : firstByte > 191 ? 2 : 1;
      if (i + bytesPerSequence <= end) {
        let secondByte, thirdByte, fourthByte, tempCodePoint;
        switch (bytesPerSequence) {
          case 1:
            if (firstByte < 128) {
              codePoint = firstByte;
            }
            break;
          case 2:
            secondByte = buf[i + 1];
            if ((secondByte & 192) === 128) {
              tempCodePoint = (firstByte & 31) << 6 | secondByte & 63;
              if (tempCodePoint > 127) {
                codePoint = tempCodePoint;
              }
            }
            break;
          case 3:
            secondByte = buf[i + 1];
            thirdByte = buf[i + 2];
            if ((secondByte & 192) === 128 && (thirdByte & 192) === 128) {
              tempCodePoint = (firstByte & 15) << 12 | (secondByte & 63) << 6 | thirdByte & 63;
              if (tempCodePoint > 2047 && (tempCodePoint < 55296 || tempCodePoint > 57343)) {
                codePoint = tempCodePoint;
              }
            }
            break;
          case 4:
            secondByte = buf[i + 1];
            thirdByte = buf[i + 2];
            fourthByte = buf[i + 3];
            if ((secondByte & 192) === 128 && (thirdByte & 192) === 128 && (fourthByte & 192) === 128) {
              tempCodePoint = (firstByte & 15) << 18 | (secondByte & 63) << 12 | (thirdByte & 63) << 6 | fourthByte & 63;
              if (tempCodePoint > 65535 && tempCodePoint < 1114112) {
                codePoint = tempCodePoint;
              }
            }
        }
      }
      if (codePoint === null) {
        codePoint = 65533;
        bytesPerSequence = 1;
      } else if (codePoint > 65535) {
        codePoint -= 65536;
        res.push(codePoint >>> 10 & 1023 | 55296);
        codePoint = 56320 | codePoint & 1023;
      }
      res.push(codePoint);
      i += bytesPerSequence;
    }
    return decodeCodePointsArray(res);
  }
  const MAX_ARGUMENTS_LENGTH = 4096;
  function decodeCodePointsArray(codePoints) {
    const len = codePoints.length;
    if (len <= MAX_ARGUMENTS_LENGTH) {
      return String.fromCharCode.apply(String, codePoints);
    }
    let res = "";
    let i = 0;
    while (i < len) {
      res += String.fromCharCode.apply(String, codePoints.slice(i, i += MAX_ARGUMENTS_LENGTH));
    }
    return res;
  }
  function asciiSlice(buf, start, end) {
    let ret = "";
    end = Math.min(buf.length, end);
    for (let i = start; i < end; ++i) {
      ret += String.fromCharCode(buf[i] & 127);
    }
    return ret;
  }
  function latin1Slice(buf, start, end) {
    let ret = "";
    end = Math.min(buf.length, end);
    for (let i = start; i < end; ++i) {
      ret += String.fromCharCode(buf[i]);
    }
    return ret;
  }
  function hexSlice(buf, start, end) {
    const len = buf.length;
    if (!start || start < 0)
      start = 0;
    if (!end || end < 0 || end > len)
      end = len;
    let out = "";
    for (let i = start; i < end; ++i) {
      out += hexSliceLookupTable[buf[i]];
    }
    return out;
  }
  function utf16leSlice(buf, start, end) {
    const bytes = buf.slice(start, end);
    let res = "";
    for (let i = 0; i < bytes.length - 1; i += 2) {
      res += String.fromCharCode(bytes[i] + bytes[i + 1] * 256);
    }
    return res;
  }
  Buffer22.prototype.slice = function slice(start, end) {
    const len = this.length;
    start = ~~start;
    end = end === void 0 ? len : ~~end;
    if (start < 0) {
      start += len;
      if (start < 0)
        start = 0;
    } else if (start > len) {
      start = len;
    }
    if (end < 0) {
      end += len;
      if (end < 0)
        end = 0;
    } else if (end > len) {
      end = len;
    }
    if (end < start)
      end = start;
    const newBuf = this.subarray(start, end);
    Object.setPrototypeOf(newBuf, Buffer22.prototype);
    return newBuf;
  };
  function checkOffset(offset, ext, length) {
    if (offset % 1 !== 0 || offset < 0)
      throw new RangeError("offset is not uint");
    if (offset + ext > length)
      throw new RangeError("Trying to access beyond buffer length");
  }
  Buffer22.prototype.readUintLE = Buffer22.prototype.readUIntLE = function readUIntLE(offset, byteLength2, noAssert) {
    offset = offset >>> 0;
    byteLength2 = byteLength2 >>> 0;
    if (!noAssert)
      checkOffset(offset, byteLength2, this.length);
    let val = this[offset];
    let mul = 1;
    let i = 0;
    while (++i < byteLength2 && (mul *= 256)) {
      val += this[offset + i] * mul;
    }
    return val;
  };
  Buffer22.prototype.readUintBE = Buffer22.prototype.readUIntBE = function readUIntBE(offset, byteLength2, noAssert) {
    offset = offset >>> 0;
    byteLength2 = byteLength2 >>> 0;
    if (!noAssert) {
      checkOffset(offset, byteLength2, this.length);
    }
    let val = this[offset + --byteLength2];
    let mul = 1;
    while (byteLength2 > 0 && (mul *= 256)) {
      val += this[offset + --byteLength2] * mul;
    }
    return val;
  };
  Buffer22.prototype.readUint8 = Buffer22.prototype.readUInt8 = function readUInt8(offset, noAssert) {
    offset = offset >>> 0;
    if (!noAssert)
      checkOffset(offset, 1, this.length);
    return this[offset];
  };
  Buffer22.prototype.readUint16LE = Buffer22.prototype.readUInt16LE = function readUInt16LE(offset, noAssert) {
    offset = offset >>> 0;
    if (!noAssert)
      checkOffset(offset, 2, this.length);
    return this[offset] | this[offset + 1] << 8;
  };
  Buffer22.prototype.readUint16BE = Buffer22.prototype.readUInt16BE = function readUInt16BE(offset, noAssert) {
    offset = offset >>> 0;
    if (!noAssert)
      checkOffset(offset, 2, this.length);
    return this[offset] << 8 | this[offset + 1];
  };
  Buffer22.prototype.readUint32LE = Buffer22.prototype.readUInt32LE = function readUInt32LE(offset, noAssert) {
    offset = offset >>> 0;
    if (!noAssert)
      checkOffset(offset, 4, this.length);
    return (this[offset] | this[offset + 1] << 8 | this[offset + 2] << 16) + this[offset + 3] * 16777216;
  };
  Buffer22.prototype.readUint32BE = Buffer22.prototype.readUInt32BE = function readUInt32BE(offset, noAssert) {
    offset = offset >>> 0;
    if (!noAssert)
      checkOffset(offset, 4, this.length);
    return this[offset] * 16777216 + (this[offset + 1] << 16 | this[offset + 2] << 8 | this[offset + 3]);
  };
  Buffer22.prototype.readBigUInt64LE = defineBigIntMethod(function readBigUInt64LE(offset) {
    offset = offset >>> 0;
    validateNumber(offset, "offset");
    const first = this[offset];
    const last = this[offset + 7];
    if (first === void 0 || last === void 0) {
      boundsError(offset, this.length - 8);
    }
    const lo = first + this[++offset] * 2 ** 8 + this[++offset] * 2 ** 16 + this[++offset] * 2 ** 24;
    const hi = this[++offset] + this[++offset] * 2 ** 8 + this[++offset] * 2 ** 16 + last * 2 ** 24;
    return BigInt(lo) + (BigInt(hi) << BigInt(32));
  });
  Buffer22.prototype.readBigUInt64BE = defineBigIntMethod(function readBigUInt64BE(offset) {
    offset = offset >>> 0;
    validateNumber(offset, "offset");
    const first = this[offset];
    const last = this[offset + 7];
    if (first === void 0 || last === void 0) {
      boundsError(offset, this.length - 8);
    }
    const hi = first * 2 ** 24 + this[++offset] * 2 ** 16 + this[++offset] * 2 ** 8 + this[++offset];
    const lo = this[++offset] * 2 ** 24 + this[++offset] * 2 ** 16 + this[++offset] * 2 ** 8 + last;
    return (BigInt(hi) << BigInt(32)) + BigInt(lo);
  });
  Buffer22.prototype.readIntLE = function readIntLE(offset, byteLength2, noAssert) {
    offset = offset >>> 0;
    byteLength2 = byteLength2 >>> 0;
    if (!noAssert)
      checkOffset(offset, byteLength2, this.length);
    let val = this[offset];
    let mul = 1;
    let i = 0;
    while (++i < byteLength2 && (mul *= 256)) {
      val += this[offset + i] * mul;
    }
    mul *= 128;
    if (val >= mul)
      val -= Math.pow(2, 8 * byteLength2);
    return val;
  };
  Buffer22.prototype.readIntBE = function readIntBE(offset, byteLength2, noAssert) {
    offset = offset >>> 0;
    byteLength2 = byteLength2 >>> 0;
    if (!noAssert)
      checkOffset(offset, byteLength2, this.length);
    let i = byteLength2;
    let mul = 1;
    let val = this[offset + --i];
    while (i > 0 && (mul *= 256)) {
      val += this[offset + --i] * mul;
    }
    mul *= 128;
    if (val >= mul)
      val -= Math.pow(2, 8 * byteLength2);
    return val;
  };
  Buffer22.prototype.readInt8 = function readInt8(offset, noAssert) {
    offset = offset >>> 0;
    if (!noAssert)
      checkOffset(offset, 1, this.length);
    if (!(this[offset] & 128))
      return this[offset];
    return (255 - this[offset] + 1) * -1;
  };
  Buffer22.prototype.readInt16LE = function readInt16LE(offset, noAssert) {
    offset = offset >>> 0;
    if (!noAssert)
      checkOffset(offset, 2, this.length);
    const val = this[offset] | this[offset + 1] << 8;
    return val & 32768 ? val | 4294901760 : val;
  };
  Buffer22.prototype.readInt16BE = function readInt16BE(offset, noAssert) {
    offset = offset >>> 0;
    if (!noAssert)
      checkOffset(offset, 2, this.length);
    const val = this[offset + 1] | this[offset] << 8;
    return val & 32768 ? val | 4294901760 : val;
  };
  Buffer22.prototype.readInt32LE = function readInt32LE(offset, noAssert) {
    offset = offset >>> 0;
    if (!noAssert)
      checkOffset(offset, 4, this.length);
    return this[offset] | this[offset + 1] << 8 | this[offset + 2] << 16 | this[offset + 3] << 24;
  };
  Buffer22.prototype.readInt32BE = function readInt32BE(offset, noAssert) {
    offset = offset >>> 0;
    if (!noAssert)
      checkOffset(offset, 4, this.length);
    return this[offset] << 24 | this[offset + 1] << 16 | this[offset + 2] << 8 | this[offset + 3];
  };
  Buffer22.prototype.readBigInt64LE = defineBigIntMethod(function readBigInt64LE(offset) {
    offset = offset >>> 0;
    validateNumber(offset, "offset");
    const first = this[offset];
    const last = this[offset + 7];
    if (first === void 0 || last === void 0) {
      boundsError(offset, this.length - 8);
    }
    const val = this[offset + 4] + this[offset + 5] * 2 ** 8 + this[offset + 6] * 2 ** 16 + (last << 24);
    return (BigInt(val) << BigInt(32)) + BigInt(first + this[++offset] * 2 ** 8 + this[++offset] * 2 ** 16 + this[++offset] * 2 ** 24);
  });
  Buffer22.prototype.readBigInt64BE = defineBigIntMethod(function readBigInt64BE(offset) {
    offset = offset >>> 0;
    validateNumber(offset, "offset");
    const first = this[offset];
    const last = this[offset + 7];
    if (first === void 0 || last === void 0) {
      boundsError(offset, this.length - 8);
    }
    const val = (first << 24) + // Overflow
    this[++offset] * 2 ** 16 + this[++offset] * 2 ** 8 + this[++offset];
    return (BigInt(val) << BigInt(32)) + BigInt(this[++offset] * 2 ** 24 + this[++offset] * 2 ** 16 + this[++offset] * 2 ** 8 + last);
  });
  Buffer22.prototype.readFloatLE = function readFloatLE(offset, noAssert) {
    offset = offset >>> 0;
    if (!noAssert)
      checkOffset(offset, 4, this.length);
    return ieee754.read(this, offset, true, 23, 4);
  };
  Buffer22.prototype.readFloatBE = function readFloatBE(offset, noAssert) {
    offset = offset >>> 0;
    if (!noAssert)
      checkOffset(offset, 4, this.length);
    return ieee754.read(this, offset, false, 23, 4);
  };
  Buffer22.prototype.readDoubleLE = function readDoubleLE(offset, noAssert) {
    offset = offset >>> 0;
    if (!noAssert)
      checkOffset(offset, 8, this.length);
    return ieee754.read(this, offset, true, 52, 8);
  };
  Buffer22.prototype.readDoubleBE = function readDoubleBE(offset, noAssert) {
    offset = offset >>> 0;
    if (!noAssert)
      checkOffset(offset, 8, this.length);
    return ieee754.read(this, offset, false, 52, 8);
  };
  function checkInt(buf, value, offset, ext, max, min) {
    if (!Buffer22.isBuffer(buf))
      throw new TypeError('"buffer" argument must be a Buffer instance');
    if (value > max || value < min)
      throw new RangeError('"value" argument is out of bounds');
    if (offset + ext > buf.length)
      throw new RangeError("Index out of range");
  }
  Buffer22.prototype.writeUintLE = Buffer22.prototype.writeUIntLE = function writeUIntLE(value, offset, byteLength2, noAssert) {
    value = +value;
    offset = offset >>> 0;
    byteLength2 = byteLength2 >>> 0;
    if (!noAssert) {
      const maxBytes = Math.pow(2, 8 * byteLength2) - 1;
      checkInt(this, value, offset, byteLength2, maxBytes, 0);
    }
    let mul = 1;
    let i = 0;
    this[offset] = value & 255;
    while (++i < byteLength2 && (mul *= 256)) {
      this[offset + i] = value / mul & 255;
    }
    return offset + byteLength2;
  };
  Buffer22.prototype.writeUintBE = Buffer22.prototype.writeUIntBE = function writeUIntBE(value, offset, byteLength2, noAssert) {
    value = +value;
    offset = offset >>> 0;
    byteLength2 = byteLength2 >>> 0;
    if (!noAssert) {
      const maxBytes = Math.pow(2, 8 * byteLength2) - 1;
      checkInt(this, value, offset, byteLength2, maxBytes, 0);
    }
    let i = byteLength2 - 1;
    let mul = 1;
    this[offset + i] = value & 255;
    while (--i >= 0 && (mul *= 256)) {
      this[offset + i] = value / mul & 255;
    }
    return offset + byteLength2;
  };
  Buffer22.prototype.writeUint8 = Buffer22.prototype.writeUInt8 = function writeUInt8(value, offset, noAssert) {
    value = +value;
    offset = offset >>> 0;
    if (!noAssert)
      checkInt(this, value, offset, 1, 255, 0);
    this[offset] = value & 255;
    return offset + 1;
  };
  Buffer22.prototype.writeUint16LE = Buffer22.prototype.writeUInt16LE = function writeUInt16LE(value, offset, noAssert) {
    value = +value;
    offset = offset >>> 0;
    if (!noAssert)
      checkInt(this, value, offset, 2, 65535, 0);
    this[offset] = value & 255;
    this[offset + 1] = value >>> 8;
    return offset + 2;
  };
  Buffer22.prototype.writeUint16BE = Buffer22.prototype.writeUInt16BE = function writeUInt16BE(value, offset, noAssert) {
    value = +value;
    offset = offset >>> 0;
    if (!noAssert)
      checkInt(this, value, offset, 2, 65535, 0);
    this[offset] = value >>> 8;
    this[offset + 1] = value & 255;
    return offset + 2;
  };
  Buffer22.prototype.writeUint32LE = Buffer22.prototype.writeUInt32LE = function writeUInt32LE(value, offset, noAssert) {
    value = +value;
    offset = offset >>> 0;
    if (!noAssert)
      checkInt(this, value, offset, 4, 4294967295, 0);
    this[offset + 3] = value >>> 24;
    this[offset + 2] = value >>> 16;
    this[offset + 1] = value >>> 8;
    this[offset] = value & 255;
    return offset + 4;
  };
  Buffer22.prototype.writeUint32BE = Buffer22.prototype.writeUInt32BE = function writeUInt32BE(value, offset, noAssert) {
    value = +value;
    offset = offset >>> 0;
    if (!noAssert)
      checkInt(this, value, offset, 4, 4294967295, 0);
    this[offset] = value >>> 24;
    this[offset + 1] = value >>> 16;
    this[offset + 2] = value >>> 8;
    this[offset + 3] = value & 255;
    return offset + 4;
  };
  function wrtBigUInt64LE(buf, value, offset, min, max) {
    checkIntBI(value, min, max, buf, offset, 7);
    let lo = Number(value & BigInt(4294967295));
    buf[offset++] = lo;
    lo = lo >> 8;
    buf[offset++] = lo;
    lo = lo >> 8;
    buf[offset++] = lo;
    lo = lo >> 8;
    buf[offset++] = lo;
    let hi = Number(value >> BigInt(32) & BigInt(4294967295));
    buf[offset++] = hi;
    hi = hi >> 8;
    buf[offset++] = hi;
    hi = hi >> 8;
    buf[offset++] = hi;
    hi = hi >> 8;
    buf[offset++] = hi;
    return offset;
  }
  function wrtBigUInt64BE(buf, value, offset, min, max) {
    checkIntBI(value, min, max, buf, offset, 7);
    let lo = Number(value & BigInt(4294967295));
    buf[offset + 7] = lo;
    lo = lo >> 8;
    buf[offset + 6] = lo;
    lo = lo >> 8;
    buf[offset + 5] = lo;
    lo = lo >> 8;
    buf[offset + 4] = lo;
    let hi = Number(value >> BigInt(32) & BigInt(4294967295));
    buf[offset + 3] = hi;
    hi = hi >> 8;
    buf[offset + 2] = hi;
    hi = hi >> 8;
    buf[offset + 1] = hi;
    hi = hi >> 8;
    buf[offset] = hi;
    return offset + 8;
  }
  Buffer22.prototype.writeBigUInt64LE = defineBigIntMethod(function writeBigUInt64LE(value, offset = 0) {
    return wrtBigUInt64LE(this, value, offset, BigInt(0), BigInt("0xffffffffffffffff"));
  });
  Buffer22.prototype.writeBigUInt64BE = defineBigIntMethod(function writeBigUInt64BE(value, offset = 0) {
    return wrtBigUInt64BE(this, value, offset, BigInt(0), BigInt("0xffffffffffffffff"));
  });
  Buffer22.prototype.writeIntLE = function writeIntLE(value, offset, byteLength2, noAssert) {
    value = +value;
    offset = offset >>> 0;
    if (!noAssert) {
      const limit = Math.pow(2, 8 * byteLength2 - 1);
      checkInt(this, value, offset, byteLength2, limit - 1, -limit);
    }
    let i = 0;
    let mul = 1;
    let sub = 0;
    this[offset] = value & 255;
    while (++i < byteLength2 && (mul *= 256)) {
      if (value < 0 && sub === 0 && this[offset + i - 1] !== 0) {
        sub = 1;
      }
      this[offset + i] = (value / mul >> 0) - sub & 255;
    }
    return offset + byteLength2;
  };
  Buffer22.prototype.writeIntBE = function writeIntBE(value, offset, byteLength2, noAssert) {
    value = +value;
    offset = offset >>> 0;
    if (!noAssert) {
      const limit = Math.pow(2, 8 * byteLength2 - 1);
      checkInt(this, value, offset, byteLength2, limit - 1, -limit);
    }
    let i = byteLength2 - 1;
    let mul = 1;
    let sub = 0;
    this[offset + i] = value & 255;
    while (--i >= 0 && (mul *= 256)) {
      if (value < 0 && sub === 0 && this[offset + i + 1] !== 0) {
        sub = 1;
      }
      this[offset + i] = (value / mul >> 0) - sub & 255;
    }
    return offset + byteLength2;
  };
  Buffer22.prototype.writeInt8 = function writeInt8(value, offset, noAssert) {
    value = +value;
    offset = offset >>> 0;
    if (!noAssert)
      checkInt(this, value, offset, 1, 127, -128);
    if (value < 0)
      value = 255 + value + 1;
    this[offset] = value & 255;
    return offset + 1;
  };
  Buffer22.prototype.writeInt16LE = function writeInt16LE(value, offset, noAssert) {
    value = +value;
    offset = offset >>> 0;
    if (!noAssert)
      checkInt(this, value, offset, 2, 32767, -32768);
    this[offset] = value & 255;
    this[offset + 1] = value >>> 8;
    return offset + 2;
  };
  Buffer22.prototype.writeInt16BE = function writeInt16BE(value, offset, noAssert) {
    value = +value;
    offset = offset >>> 0;
    if (!noAssert)
      checkInt(this, value, offset, 2, 32767, -32768);
    this[offset] = value >>> 8;
    this[offset + 1] = value & 255;
    return offset + 2;
  };
  Buffer22.prototype.writeInt32LE = function writeInt32LE(value, offset, noAssert) {
    value = +value;
    offset = offset >>> 0;
    if (!noAssert)
      checkInt(this, value, offset, 4, 2147483647, -2147483648);
    this[offset] = value & 255;
    this[offset + 1] = value >>> 8;
    this[offset + 2] = value >>> 16;
    this[offset + 3] = value >>> 24;
    return offset + 4;
  };
  Buffer22.prototype.writeInt32BE = function writeInt32BE(value, offset, noAssert) {
    value = +value;
    offset = offset >>> 0;
    if (!noAssert)
      checkInt(this, value, offset, 4, 2147483647, -2147483648);
    if (value < 0)
      value = 4294967295 + value + 1;
    this[offset] = value >>> 24;
    this[offset + 1] = value >>> 16;
    this[offset + 2] = value >>> 8;
    this[offset + 3] = value & 255;
    return offset + 4;
  };
  Buffer22.prototype.writeBigInt64LE = defineBigIntMethod(function writeBigInt64LE(value, offset = 0) {
    return wrtBigUInt64LE(this, value, offset, -BigInt("0x8000000000000000"), BigInt("0x7fffffffffffffff"));
  });
  Buffer22.prototype.writeBigInt64BE = defineBigIntMethod(function writeBigInt64BE(value, offset = 0) {
    return wrtBigUInt64BE(this, value, offset, -BigInt("0x8000000000000000"), BigInt("0x7fffffffffffffff"));
  });
  function checkIEEE754(buf, value, offset, ext, max, min) {
    if (offset + ext > buf.length)
      throw new RangeError("Index out of range");
    if (offset < 0)
      throw new RangeError("Index out of range");
  }
  function writeFloat(buf, value, offset, littleEndian, noAssert) {
    value = +value;
    offset = offset >>> 0;
    if (!noAssert) {
      checkIEEE754(buf, value, offset, 4);
    }
    ieee754.write(buf, value, offset, littleEndian, 23, 4);
    return offset + 4;
  }
  Buffer22.prototype.writeFloatLE = function writeFloatLE(value, offset, noAssert) {
    return writeFloat(this, value, offset, true, noAssert);
  };
  Buffer22.prototype.writeFloatBE = function writeFloatBE(value, offset, noAssert) {
    return writeFloat(this, value, offset, false, noAssert);
  };
  function writeDouble(buf, value, offset, littleEndian, noAssert) {
    value = +value;
    offset = offset >>> 0;
    if (!noAssert) {
      checkIEEE754(buf, value, offset, 8);
    }
    ieee754.write(buf, value, offset, littleEndian, 52, 8);
    return offset + 8;
  }
  Buffer22.prototype.writeDoubleLE = function writeDoubleLE(value, offset, noAssert) {
    return writeDouble(this, value, offset, true, noAssert);
  };
  Buffer22.prototype.writeDoubleBE = function writeDoubleBE(value, offset, noAssert) {
    return writeDouble(this, value, offset, false, noAssert);
  };
  Buffer22.prototype.copy = function copy(target, targetStart, start, end) {
    if (!Buffer22.isBuffer(target))
      throw new TypeError("argument should be a Buffer");
    if (!start)
      start = 0;
    if (!end && end !== 0)
      end = this.length;
    if (targetStart >= target.length)
      targetStart = target.length;
    if (!targetStart)
      targetStart = 0;
    if (end > 0 && end < start)
      end = start;
    if (end === start)
      return 0;
    if (target.length === 0 || this.length === 0)
      return 0;
    if (targetStart < 0) {
      throw new RangeError("targetStart out of bounds");
    }
    if (start < 0 || start >= this.length)
      throw new RangeError("Index out of range");
    if (end < 0)
      throw new RangeError("sourceEnd out of bounds");
    if (end > this.length)
      end = this.length;
    if (target.length - targetStart < end - start) {
      end = target.length - targetStart + start;
    }
    const len = end - start;
    if (this === target && typeof Uint8Array.prototype.copyWithin === "function") {
      this.copyWithin(targetStart, start, end);
    } else {
      Uint8Array.prototype.set.call(target, this.subarray(start, end), targetStart);
    }
    return len;
  };
  Buffer22.prototype.fill = function fill(val, start, end, encoding) {
    if (typeof val === "string") {
      if (typeof start === "string") {
        encoding = start;
        start = 0;
        end = this.length;
      } else if (typeof end === "string") {
        encoding = end;
        end = this.length;
      }
      if (encoding !== void 0 && typeof encoding !== "string") {
        throw new TypeError("encoding must be a string");
      }
      if (typeof encoding === "string" && !Buffer22.isEncoding(encoding)) {
        throw new TypeError("Unknown encoding: " + encoding);
      }
      if (val.length === 1) {
        const code2 = val.charCodeAt(0);
        if (encoding === "utf8" && code2 < 128 || encoding === "latin1") {
          val = code2;
        }
      }
    } else if (typeof val === "number") {
      val = val & 255;
    } else if (typeof val === "boolean") {
      val = Number(val);
    }
    if (start < 0 || this.length < start || this.length < end) {
      throw new RangeError("Out of range index");
    }
    if (end <= start) {
      return this;
    }
    start = start >>> 0;
    end = end === void 0 ? this.length : end >>> 0;
    if (!val)
      val = 0;
    let i;
    if (typeof val === "number") {
      for (i = start; i < end; ++i) {
        this[i] = val;
      }
    } else {
      const bytes = Buffer22.isBuffer(val) ? val : Buffer22.from(val, encoding);
      const len = bytes.length;
      if (len === 0) {
        throw new TypeError('The value "' + val + '" is invalid for argument "value"');
      }
      for (i = 0; i < end - start; ++i) {
        this[i + start] = bytes[i % len];
      }
    }
    return this;
  };
  const errors = {};
  function E(sym, getMessage, Base) {
    errors[sym] = class NodeError extends Base {
      constructor() {
        super();
        Object.defineProperty(this, "message", {
          value: getMessage.apply(this, arguments),
          writable: true,
          configurable: true
        });
        this.name = `${this.name} [${sym}]`;
        this.stack;
        delete this.name;
      }
      get code() {
        return sym;
      }
      set code(value) {
        Object.defineProperty(this, "code", {
          configurable: true,
          enumerable: true,
          value,
          writable: true
        });
      }
      toString() {
        return `${this.name} [${sym}]: ${this.message}`;
      }
    };
  }
  E("ERR_BUFFER_OUT_OF_BOUNDS", function(name) {
    if (name) {
      return `${name} is outside of buffer bounds`;
    }
    return "Attempt to access memory outside buffer bounds";
  }, RangeError);
  E("ERR_INVALID_ARG_TYPE", function(name, actual) {
    return `The "${name}" argument must be of type number. Received type ${typeof actual}`;
  }, TypeError);
  E("ERR_OUT_OF_RANGE", function(str, range, input) {
    let msg = `The value of "${str}" is out of range.`;
    let received = input;
    if (Number.isInteger(input) && Math.abs(input) > 2 ** 32) {
      received = addNumericalSeparator(String(input));
    } else if (typeof input === "bigint") {
      received = String(input);
      if (input > BigInt(2) ** BigInt(32) || input < -(BigInt(2) ** BigInt(32))) {
        received = addNumericalSeparator(received);
      }
      received += "n";
    }
    msg += ` It must be ${range}. Received ${received}`;
    return msg;
  }, RangeError);
  function addNumericalSeparator(val) {
    let res = "";
    let i = val.length;
    const start = val[0] === "-" ? 1 : 0;
    for (; i >= start + 4; i -= 3) {
      res = `_${val.slice(i - 3, i)}${res}`;
    }
    return `${val.slice(0, i)}${res}`;
  }
  function checkBounds(buf, offset, byteLength2) {
    validateNumber(offset, "offset");
    if (buf[offset] === void 0 || buf[offset + byteLength2] === void 0) {
      boundsError(offset, buf.length - (byteLength2 + 1));
    }
  }
  function checkIntBI(value, min, max, buf, offset, byteLength2) {
    if (value > max || value < min) {
      const n = typeof min === "bigint" ? "n" : "";
      let range;
      {
        if (min === 0 || min === BigInt(0)) {
          range = `>= 0${n} and < 2${n} ** ${(byteLength2 + 1) * 8}${n}`;
        } else {
          range = `>= -(2${n} ** ${(byteLength2 + 1) * 8 - 1}${n}) and < 2 ** ${(byteLength2 + 1) * 8 - 1}${n}`;
        }
      }
      throw new errors.ERR_OUT_OF_RANGE("value", range, value);
    }
    checkBounds(buf, offset, byteLength2);
  }
  function validateNumber(value, name) {
    if (typeof value !== "number") {
      throw new errors.ERR_INVALID_ARG_TYPE(name, "number", value);
    }
  }
  function boundsError(value, length, type) {
    if (Math.floor(value) !== value) {
      validateNumber(value, type);
      throw new errors.ERR_OUT_OF_RANGE("offset", "an integer", value);
    }
    if (length < 0) {
      throw new errors.ERR_BUFFER_OUT_OF_BOUNDS();
    }
    throw new errors.ERR_OUT_OF_RANGE("offset", `>= ${0} and <= ${length}`, value);
  }
  const INVALID_BASE64_RE = /[^+/0-9A-Za-z-_]/g;
  function base64clean(str) {
    str = str.split("=")[0];
    str = str.trim().replace(INVALID_BASE64_RE, "");
    if (str.length < 2)
      return "";
    while (str.length % 4 !== 0) {
      str = str + "=";
    }
    return str;
  }
  function utf8ToBytes(string, units) {
    units = units || Infinity;
    let codePoint;
    const length = string.length;
    let leadSurrogate = null;
    const bytes = [];
    for (let i = 0; i < length; ++i) {
      codePoint = string.charCodeAt(i);
      if (codePoint > 55295 && codePoint < 57344) {
        if (!leadSurrogate) {
          if (codePoint > 56319) {
            if ((units -= 3) > -1)
              bytes.push(239, 191, 189);
            continue;
          } else if (i + 1 === length) {
            if ((units -= 3) > -1)
              bytes.push(239, 191, 189);
            continue;
          }
          leadSurrogate = codePoint;
          continue;
        }
        if (codePoint < 56320) {
          if ((units -= 3) > -1)
            bytes.push(239, 191, 189);
          leadSurrogate = codePoint;
          continue;
        }
        codePoint = (leadSurrogate - 55296 << 10 | codePoint - 56320) + 65536;
      } else if (leadSurrogate) {
        if ((units -= 3) > -1)
          bytes.push(239, 191, 189);
      }
      leadSurrogate = null;
      if (codePoint < 128) {
        if ((units -= 1) < 0)
          break;
        bytes.push(codePoint);
      } else if (codePoint < 2048) {
        if ((units -= 2) < 0)
          break;
        bytes.push(codePoint >> 6 | 192, codePoint & 63 | 128);
      } else if (codePoint < 65536) {
        if ((units -= 3) < 0)
          break;
        bytes.push(codePoint >> 12 | 224, codePoint >> 6 & 63 | 128, codePoint & 63 | 128);
      } else if (codePoint < 1114112) {
        if ((units -= 4) < 0)
          break;
        bytes.push(codePoint >> 18 | 240, codePoint >> 12 & 63 | 128, codePoint >> 6 & 63 | 128, codePoint & 63 | 128);
      } else {
        throw new Error("Invalid code point");
      }
    }
    return bytes;
  }
  function asciiToBytes(str) {
    const byteArray = [];
    for (let i = 0; i < str.length; ++i) {
      byteArray.push(str.charCodeAt(i) & 255);
    }
    return byteArray;
  }
  function utf16leToBytes(str, units) {
    let c, hi, lo;
    const byteArray = [];
    for (let i = 0; i < str.length; ++i) {
      if ((units -= 2) < 0)
        break;
      c = str.charCodeAt(i);
      hi = c >> 8;
      lo = c % 256;
      byteArray.push(lo);
      byteArray.push(hi);
    }
    return byteArray;
  }
  function base64ToBytes(str) {
    return base64.toByteArray(base64clean(str));
  }
  function blitBuffer(src, dst, offset, length) {
    let i;
    for (i = 0; i < length; ++i) {
      if (i + offset >= dst.length || i >= src.length)
        break;
      dst[i + offset] = src[i];
    }
    return i;
  }
  function isInstance(obj, type) {
    return obj instanceof type || obj != null && obj.constructor != null && obj.constructor.name != null && obj.constructor.name === type.name;
  }
  function numberIsNaN(obj) {
    return obj !== obj;
  }
  const hexSliceLookupTable = function() {
    const alphabet = "0123456789abcdef";
    const table2 = new Array(256);
    for (let i = 0; i < 16; ++i) {
      const i16 = i * 16;
      for (let j = 0; j < 16; ++j) {
        table2[i16 + j] = alphabet[i] + alphabet[j];
      }
    }
    return table2;
  }();
  function defineBigIntMethod(fn) {
    return typeof BigInt === "undefined" ? BufferBigIntNotDefined : fn;
  }
  function BufferBigIntNotDefined() {
    throw new Error("BigInt not supported");
  }
  return exports;
}
var exports2 = dew();
exports2["Buffer"];
exports2["SlowBuffer"];
exports2["INSPECT_MAX_BYTES"];
exports2["kMaxLength"];
var Buffer2 = exports2.Buffer;
var INSPECT_MAX_BYTES = exports2.INSPECT_MAX_BYTES;
var kMaxLength = exports2.kMaxLength;

// ../../node_modules/myst-to-docx/dist/utils.js
function createShortId() {
  return Math.random().toString(36).slice(2);
}
function createDocFromState(state, footer, styles2) {
  const { title, description, keywords } = state.frontmatter;
  const doc = new import_docx.Document({
    title,
    description,
    keywords: keywords === null || keywords === void 0 ? void 0 : keywords.join(", "),
    footnotes: state.footnotes,
    numbering: {
      config: state.numbering
    },
    sections: [
      {
        properties: {
          type: import_docx.SectionType.CONTINUOUS
        },
        children: state.children,
        footers: footer ? { default: footer } : void 0
      }
    ],
    externalStyles: styles2
  });
  return doc;
}
async function writeDocx(doc, write) {
  const buffer = await import_docx.Packer.toBuffer(doc);
  return write(buffer);
}
var DEFAULT_IMAGE_WIDTH = 70;
var DEFAULT_PAGE_WIDTH_PIXELS = 800;
var MAX_DOCX_IMAGE_WIDTH = 600;
function getImageWidth(width, maxWidth = MAX_DOCX_IMAGE_WIDTH) {
  if (typeof width === "number" && Number.isNaN(width)) {
    return getImageWidth(DEFAULT_IMAGE_WIDTH);
  }
  if (typeof width === "string") {
    if (width.endsWith("%")) {
      return getImageWidth(Number(width.replace("%", "")));
    } else if (width.endsWith("px")) {
      return getImageWidth(Number(width.replace("px", "")) / DEFAULT_PAGE_WIDTH_PIXELS);
    }
    console.log(`Unknown width ${width} in getImageWidth`);
    return getImageWidth(DEFAULT_IMAGE_WIDTH);
  }
  let lineWidth = width !== null && width !== void 0 ? width : DEFAULT_IMAGE_WIDTH;
  if (lineWidth < 1)
    lineWidth *= 100;
  if (lineWidth > 100)
    lineWidth = 100;
  return lineWidth / 100 * maxWidth;
}
async function getImageDimensions(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const { naturalWidth: width, naturalHeight: height } = img;
      resolve({ width, height });
    };
    img.onerror = () => {
      reject("There was some problem with the image.");
    };
    img.src = URL.createObjectURL(file);
  });
}
async function fetchImagesAsBuffers(tree) {
  const images = selectAll("image", tree);
  const buffers = {};
  const dimensions = {};
  await Promise.all(images.map(async (image2) => {
    const response = await fetch(image2.url);
    const blob = await response.blob();
    const buffer = await blob.arrayBuffer();
    dimensions[image2.url] = await getImageDimensions(blob);
    buffers[image2.url] = Buffer2.from(buffer);
  }));
  return {
    getImageBuffer(url) {
      return buffers[url];
    },
    getImageDimensions(url) {
      return dimensions[url];
    }
  };
}
function createReferenceBookmark(id, kind, before, after) {
  const textBefore = before ? [new import_docx.TextRun(before)] : [];
  const textAfter = after ? [new import_docx.TextRun(after)] : [];
  return new import_docx.Bookmark({
    id,
    children: [...textBefore, new import_docx.SequentialIdentifier(kind), ...textAfter]
  });
}
function createReference(id, before, after) {
  const children = [];
  if (before)
    children.push(new import_docx.TextRun(before));
  children.push(new import_docx.SimpleField(`REF ${id} \\h`));
  if (after)
    children.push(new import_docx.TextRun(after));
  const ref = new import_docx.InternalHyperlink({ anchor: id, children });
  return ref;
}

// ../../node_modules/myst-to-docx/dist/numbering.js
var import_docx2 = __toESM(require_build(), 1);
function basicIndentStyle(indent) {
  return {
    alignment: import_docx2.AlignmentType.START,
    style: {
      paragraph: {
        indent: { left: (0, import_docx2.convertInchesToTwip)(indent), hanging: (0, import_docx2.convertInchesToTwip)(0.18) }
      }
    }
  };
}
var numbered = Array(3).fill([import_docx2.LevelFormat.DECIMAL, import_docx2.LevelFormat.LOWER_LETTER, import_docx2.LevelFormat.LOWER_ROMAN]).flat().map((format, level) => ({
  level,
  format,
  text: `%${level + 1}.`,
  ...basicIndentStyle((level + 1) / 2)
}));
var bullets = Array(3).fill(["\u25CF", "\u25CB", "\u25A0"]).flat().map((text2, level) => ({
  level,
  format: import_docx2.LevelFormat.BULLET,
  text: text2,
  ...basicIndentStyle((level + 1) / 2)
}));
var styles = {
  numbered,
  bullets
};
function createNumbering(reference, style) {
  return {
    reference,
    levels: styles[style]
  };
}

// ../../node_modules/myst-to-docx/dist/schema.js
var import_buffer_image_size = __toESM(require_lib(), 1);
var text = (state, node) => {
  var _a;
  state.text((_a = node.value) !== null && _a !== void 0 ? _a : "");
};
var paragraph = (state, node) => {
  state.renderChildren(node);
  state.closeBlock();
};
var block = (state, node) => {
  if (node.visibility === "remove" || node.visibility === "hide")
    return;
  const metadataTags = getMetadataTags(node);
  if (metadataTags.includes("page-break") || metadataTags.includes("new-page")) {
    state.current.push(new import_docx3.PageBreak());
  }
  state.renderChildren(node);
};
var heading = (state, node) => {
  if (!state.options.useFieldsForCrossReferences && node.enumerator) {
    state.text(`${node.enumerator}	`);
  } else {
  }
  state.renderChildren(node);
  const headingLevel = [
    import_docx3.HeadingLevel.HEADING_1,
    import_docx3.HeadingLevel.HEADING_2,
    import_docx3.HeadingLevel.HEADING_3,
    import_docx3.HeadingLevel.HEADING_4,
    import_docx3.HeadingLevel.HEADING_5,
    import_docx3.HeadingLevel.HEADING_6
  ][node.depth - 1];
  state.closeBlock({ heading: headingLevel });
};
var emphasis = (state, node) => {
  state.addRunOptions({ italics: true });
  state.renderChildren(node);
};
var strong = (state, node) => {
  state.addRunOptions({ bold: true });
  state.renderChildren(node);
};
var list = (state, node) => {
  const style = node.ordered ? "numbered" : "bullets";
  if (!state.data.currentNumbering) {
    const nextId = createShortId();
    state.numbering.push(createNumbering(nextId, style));
    state.data.currentNumbering = { reference: nextId, level: 0 };
  } else {
    const { reference, level } = state.data.currentNumbering;
    state.data.currentNumbering = { reference, level: level + 1 };
  }
  state.renderChildren(node);
  if (state.data.currentNumbering.level === 0) {
    delete state.data.currentNumbering;
  } else {
    const { reference, level } = state.data.currentNumbering;
    state.data.currentNumbering = { reference, level: level - 1 };
  }
};
var listItem = (state, node, parent) => {
  if (!state.data.currentNumbering)
    throw new Error("Trying to create a list item without a list?");
  if (state.current.length > 0) {
    state.closeBlock();
  }
  state.addParagraphOptions({ numbering: state.data.currentNumbering });
  state.renderChildren(node);
  if (parent.type !== "paragraph") {
    state.closeBlock();
  }
};
var link = (state, node) => {
  const stack = state.current;
  state.addRunOptions({ style: "Hyperlink" });
  state.current = [];
  state.renderChildren(node);
  const hyperlink = new import_docx3.ExternalHyperlink({
    link: node.url,
    children: state.current
  });
  state.current = [...stack, hyperlink];
};
var inlineCode = (state, node) => {
  state.text(node.value, {
    font: {
      name: "Monospace"
    },
    color: "000000",
    shading: {
      type: import_docx3.ShadingType.SOLID,
      color: "D2D3D2",
      fill: "D2D3D2"
    }
  });
};
var _break = (state) => {
  state.addRunOptions({ break: 1 });
};
var thematicBreak = (state) => {
  state.closeBlock({ thematicBreak: true });
  state.blankLine();
};
var abbreviation = (state, node) => {
  state.renderChildren(node);
};
var subscript = (state, node) => {
  state.addRunOptions({ subScript: true });
  state.renderChildren(node);
};
var superscript = (state, node) => {
  state.addRunOptions({ superScript: true });
  state.renderChildren(node);
};
var _delete = (state, node) => {
  state.addRunOptions({ strike: true });
  state.renderChildren(node);
};
var underline = (state, node) => {
  state.addRunOptions({ underline: {} });
  state.renderChildren(node);
};
var smallcaps = (state, node) => {
  state.addRunOptions({ smallCaps: true });
  state.renderChildren(node);
};
var blockquote = (state, node) => {
  state.renderChildren(node, { style: "IntenseQuote" });
};
var code = (state, node) => {
  node.value.split("\n").forEach((line) => {
    state.text(line, {
      font: {
        name: "Monospace"
      }
    });
    state.closeBlock();
  });
};
function getAspect(buffer, size) {
  if (size)
    return size.height / size.width;
  try {
    const dimensions = (0, import_buffer_image_size.default)(buffer);
    return dimensions.height / dimensions.width;
  } catch (error) {
    return void 0;
  }
}
var image = (state, node) => {
  var _a, _b, _c;
  const buffer = state.options.getImageBuffer(node.url);
  const dimensions = (_b = (_a = state.options).getImageDimensions) === null || _b === void 0 ? void 0 : _b.call(_a, node.url);
  const width = getImageWidth(node.width, (_c = state.data.maxImageWidth) !== null && _c !== void 0 ? _c : state.options.maxImageWidth);
  const aspect = getAspect(buffer, dimensions);
  if (!aspect) {
    fileError(state.file, `Error with checking image aspect ratio for "${node.url}".`, {
      node,
      source: "myst-to-docx:image",
      note: 'Either provide dimensions of the image with "getImageDimensions" or ensure that the result is a Buffer.',
      ruleId: RuleId.docxRenders
    });
  }
  state.current.push(new import_docx3.ImageRun({
    data: buffer,
    transformation: {
      width,
      height: width * (aspect !== null && aspect !== void 0 ? aspect : 1)
    }
  }));
  let alignment;
  switch (node.align) {
    case "right":
      alignment = import_docx3.AlignmentType.RIGHT;
      break;
    case "left":
      alignment = import_docx3.AlignmentType.LEFT;
      break;
    default:
      alignment = import_docx3.AlignmentType.CENTER;
  }
  state.addParagraphOptions({
    alignment
  });
  state.closeBlock();
};
var admonitionStyle = {
  border: {
    left: {
      style: import_docx3.BorderStyle.DOUBLE,
      color: "5D85D0"
    }
  },
  indent: { left: (0, import_docx3.convertInchesToTwip)(0.2), right: (0, import_docx3.convertInchesToTwip)(0.2) }
};
var admonition = (state, node) => {
  state.blankLine();
  state.renderChildren(node, admonitionStyle);
  state.closeBlock();
  state.blankLine();
};
var admonitionTitle = (state, node) => {
  state.renderChildren(node, {
    ...admonitionStyle,
    shading: {
      type: import_docx3.ShadingType.SOLID,
      color: "5D85D0"
    }
  }, { bold: true, color: "FFFFFF" });
  state.closeBlock();
};
var definitionStyle = {
  border: {
    left: {
      style: import_docx3.BorderStyle.THICK,
      color: "D2D3D2"
    }
  },
  indent: { left: (0, import_docx3.convertInchesToTwip)(0.2), right: (0, import_docx3.convertInchesToTwip)(0.2) }
};
var definitionList = (state, node) => {
  state.renderChildren(node, definitionStyle);
  state.closeBlock();
  state.blankLine();
};
var definitionTerm = (state, node) => {
  state.renderChildren(node, {
    ...definitionStyle,
    shading: {
      type: import_docx3.ShadingType.SOLID,
      color: "D2D3D2"
    }
  });
  state.closeBlock();
};
var definitionDescription = (state, node) => {
  state.text("	");
  state.renderChildren(node, definitionStyle);
  state.closeBlock();
};
var inlineMath = (state, node) => {
  const latex = node.value;
  state.current.push(new import_docx3.Math({ children: [new import_docx3.MathRun(latex)] }));
};
var math = (state, node) => {
  state.blankLine();
  const latex = node.value;
  state.current = [
    new import_docx3.TextRun("	"),
    new import_docx3.Math({
      children: [new import_docx3.MathRun(latex)]
    })
  ];
  if (node.enumerator && node.identifier && state.options.useFieldsForCrossReferences) {
    state.current.push(new import_docx3.TextRun("	("), createReferenceBookmark(node.identifier, "Equation"), new import_docx3.TextRun(")"));
  } else if (node.enumerator) {
    state.current.push(new import_docx3.TextRun(`	(${node.enumerator})`));
  }
  state.closeBlock({
    tabStops: [
      {
        type: import_docx3.TabStopType.CENTER,
        position: import_docx3.TabStopPosition.MAX / 2
      },
      {
        type: import_docx3.TabStopType.RIGHT,
        position: import_docx3.TabStopPosition.MAX
      }
    ]
  });
  state.blankLine();
};
var crossReference = (state, node) => {
  if (state.options.useFieldsForCrossReferences && node.identifier) {
    state.current.push(createReference(node.identifier));
  } else {
    state.renderChildren(node);
  }
};
var container = (state, node) => {
  state.renderChildren(node);
};
function figCaptionToWordCaption(file, kind) {
  switch (kind.toLowerCase()) {
    case "figure":
      return "Figure";
    case "table":
      return "Table";
    case "equation":
      return "Equation";
    case "code":
      return "Figure";
    default:
      fileError(file, `Unknown figure caption of kind ${kind}`, {
        ruleId: RuleId.docxRenders
      });
      return "Figure";
  }
}
var captionNumber = (state, node) => {
  if (state.options.useFieldsForCrossReferences) {
    const bookmarkKind = figCaptionToWordCaption(state.file, node.kind);
    state.current.push(createReferenceBookmark(node.identifier, bookmarkKind, `${bookmarkKind} `, ": "));
  } else {
    state.renderChildren(node, void 0, { bold: true });
    state.text(" ");
  }
};
var caption = (state, node) => {
  state.renderChildren(node, { style: "Caption" });
};
function getFootnoteNumber(node) {
  var _a;
  return (_a = Number.parseInt(node.enumerator, 10)) !== null && _a !== void 0 ? _a : Number(node.identifier);
}
var footnoteDefinition = (state, node) => {
  const { children, current } = state;
  const number = getFootnoteNumber(node);
  state.children = [];
  state.current = [];
  state.renderChildren(node);
  state.footnotes[number] = { children: state.children };
  state.children = children;
  state.current = current;
};
var footnoteReference = (state, node) => {
  const number = getFootnoteNumber(node);
  state.current.push(new import_docx3.FootnoteReferenceRun(number));
};
var table = (state, node) => {
  var _a, _b;
  const actualChildren = state.children;
  const rows = [];
  const imageWidth = (_b = (_a = state.data.maxImageWidth) !== null && _a !== void 0 ? _a : state.options.maxImageWidth) !== null && _b !== void 0 ? _b : MAX_DOCX_IMAGE_WIDTH;
  node.children.forEach(({ children }) => {
    const rowContent = children;
    const cells = [];
    let tableHeader = true;
    rowContent.forEach((cell) => {
      if (cell.header) {
        tableHeader = false;
      }
    });
    state.data.maxImageWidth = imageWidth / rowContent.length;
    rowContent.forEach((cell) => {
      var _a2, _b2;
      state.children = [];
      state.renderChildren(cell);
      state.closeBlock();
      const tableCellOpts = { children: state.children };
      const colspan = (_a2 = cell.colspan) !== null && _a2 !== void 0 ? _a2 : 1;
      const rowspan = (_b2 = cell.rowspan) !== null && _b2 !== void 0 ? _b2 : 1;
      if (colspan > 1)
        tableCellOpts.columnSpan = colspan;
      if (rowspan > 1)
        tableCellOpts.rowSpan = rowspan;
      cells.push(new import_docx3.TableCell(tableCellOpts));
    });
    rows.push(new import_docx3.TableRow({ children: cells, tableHeader }));
  });
  state.data.maxImageWidth = imageWidth;
  const tableNode = new import_docx3.Table({ rows });
  actualChildren.push(tableNode);
  actualChildren.push(new import_docx3.Paragraph(""));
  state.children = actualChildren;
};
var cite = (state, node) => {
  state.renderChildren(node);
};
var citeGroup = (state, node) => {
  if (node.kind === "narrative") {
    state.renderChildren(node);
  } else {
    state.text("(");
    node.children.forEach((child, ind) => {
      state.render(child);
      if (ind < node.children.length - 1)
        state.text("; ");
    });
    state.text(")");
  }
};
var embed = (state, node) => {
  state.renderChildren(node);
};
var include = (state, node) => {
  state.renderChildren(node);
};
var comment = () => {
  return;
};
var mystDirective = (state, node) => {
  state.renderChildren(node);
};
var mystRole = (state, node) => {
  state.renderChildren(node);
};
var inlineExpression = (state, node, parent) => {
  var _a;
  if ((_a = node.children) === null || _a === void 0 ? void 0 : _a.length) {
    state.renderChildren(node);
  } else {
    inlineCode(state, { ...node, type: "inlineCode" }, parent);
  }
};
var defaultHandlers = {
  text,
  paragraph,
  heading,
  emphasis,
  strong,
  inlineCode,
  link,
  break: _break,
  thematicBreak,
  list,
  listItem,
  abbreviation,
  subscript,
  superscript,
  delete: _delete,
  underline,
  smallcaps,
  blockquote,
  code,
  image,
  block,
  comment,
  mystDirective,
  mystRole,
  admonition,
  admonitionTitle,
  definitionList,
  definitionTerm,
  definitionDescription,
  math,
  inlineMath,
  crossReference,
  container,
  caption,
  captionNumber,
  footnoteReference,
  footnoteDefinition,
  table,
  cite,
  citeGroup,
  embed,
  include,
  inlineExpression
};

// ../../node_modules/myst-to-docx/dist/serializer.js
var DocxSerializer = class {
  constructor(file, options, frontmatter) {
    var _a;
    this.footnotes = {};
    this.current = [];
    this.frontmatter = {};
    this.file = file;
    this.data = {};
    this.handlers = (_a = options.handlers) !== null && _a !== void 0 ? _a : defaultHandlers;
    this.options = options !== null && options !== void 0 ? options : {};
    this.children = [];
    this.numbering = [];
    this.frontmatter = frontmatter !== null && frontmatter !== void 0 ? frontmatter : {};
  }
  render(node, parent) {
    if (!this.handlers[node.type]) {
      fileError(this.file, `Node of type "${node.type}" is not supported by docx renderer`, {
        node,
        source: "myst-to-docx:render",
        ruleId: RuleId.docxRenders
      });
      return;
    }
    this.handlers[node.type](this, node, parent);
  }
  renderChildren(parent, paragraphOpts, runOpts) {
    if (!("children" in parent)) {
      const node = parent;
      fileWarn(this.file, `Expected children for node of type ${node.type}`, {
        node,
        ruleId: RuleId.docxRenders
      });
      return;
    }
    parent.children.forEach((node) => {
      if (paragraphOpts)
        this.addParagraphOptions(paragraphOpts);
      if (runOpts)
        this.addRunOptions(runOpts);
      this.render(node, parent);
    });
  }
  addParagraphOptions(opts) {
    var _a;
    this.data.nextParagraphOpts = { ...(_a = this.data) === null || _a === void 0 ? void 0 : _a.nextParagraphOpts, ...opts };
  }
  addRunOptions(opts) {
    this.data.nextRunOpts = { ...this.data.nextRunOpts, ...opts };
  }
  text(text2, opts) {
    if (!text2)
      return;
    this.current.push(new import_docx4.TextRun({ text: text2, ...this.data.nextRunOpts, ...opts }));
    delete this.data.nextRunOpts;
  }
  closeBlock(props, force = false) {
    if (this.current.length === 0 && !props && !force) {
      delete this.data.nextParagraphOpts;
      return;
    }
    const paragraph2 = new import_docx4.Paragraph({
      children: this.current,
      ...this.data.nextParagraphOpts,
      ...props
    });
    this.current = [];
    delete this.data.nextParagraphOpts;
    this.children.push(paragraph2);
  }
  blankLine(props) {
    this.closeBlock(props, true);
  }
};

// ../../node_modules/myst-to-docx/dist/plugin.js
var import_docx5 = __toESM(require_build(), 1);
var plugin = function(opts) {
  this.Compiler = (node, file) => {
    const state = new DocxSerializer(file, opts);
    state.renderChildren(node);
    const doc = createDocFromState(state);
    if (typeof document === "undefined") {
      file.result = import_docx5.Packer.toBuffer(doc);
    } else {
      file.result = import_docx5.Packer.toBlob(doc);
    }
    return file;
  };
  return (node) => {
    return node;
  };
};
export {
  DocxSerializer,
  createDocFromState,
  defaultHandlers,
  fetchImagesAsBuffers,
  plugin as mystToDocx,
  writeDocx
};
/*! Bundled license information:

@jspm/core/nodelibs/browser/chunk-DtuTasat.js:
  (*! ieee754. BSD-3-Clause License. Feross Aboukhadijeh <https://feross.org/opensource> *)
*/
/*! Bundled license information:

docx/build/index.js:
  (*! For license information please see index.js.LICENSE.txt *)
*/
//# sourceMappingURL=/build/_shared/dist-5Q6NXYLZ.js.map
