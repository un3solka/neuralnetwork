function makeArray(d1, d2) {
    let arr = new Array(d1);
    for(let i = 0; i < d1; i++) {
        arr[i] = new Array(d2);
    }
    return arr;
}

function arrayCopy(src, srcIndex, dest, destIndex, length) {
    dest.splice(destIndex, length, ...src.slice(srcIndex, srcIndex + length));
}

module.exports = {
    makeArray,
    arrayCopy
}